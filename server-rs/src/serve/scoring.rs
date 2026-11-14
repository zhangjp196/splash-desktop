//! Finite-option scoring shared by the judgments and systemone routes.
//!
//! Prompts ask for one answer slot and submit those slot token IDs through
//! the engine's native score-only protocol. No generated answer is sampled.

use serde_json::{json, Map, Value};

use crate::protocol::SamplingParameters;
use crate::serve::pipeline::{GenerationFrameOptions, Pipeline, PipelineError, Prepared};
use crate::support::errors::{field_error, ApiError};

pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";
pub const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
pub const SYSTEMONE_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase slot, with no explanation or reasoning.";
pub const PROMPT_VERSION: &str = "direct-options-v1";
pub const READOUT: &str = "native full-vocabulary last-position logits restricted to declared answer slots";
pub const PROBABILITY_STATUS: &str = "conditional option score; uncalibrated as decision confidence";
pub const MAX_SYSTEMONE_QUESTIONS: usize = 64;
pub const MAX_SYSTEMONE_TOTAL_TOKENS: usize = 1 << 20;

#[derive(Debug, Clone)]
pub struct JudgmentRow {
    pub id: String,
    pub question: String,
    pub state: Value,
    pub options: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct SystemOneSpec {
    pub kind: String,
    pub instructions: Option<Value>,
    pub labels: Vec<String>,
    pub descriptions: Vec<Value>,
    pub legend: Option<Value>,
    pub deterministic: bool,
}

#[derive(Debug, Clone)]
pub struct SystemOneEntry {
    pub id: String,
    pub spec: SystemOneSpec,
}

impl JudgmentRow {
    pub fn messages(&self) -> Value {
        let evidence = serde_json::to_string(&self.state).expect("validated JSON state");
        let criterion = serde_json::to_string(&self.question).expect("JSON string");
        let options = self.options.iter().enumerate().map(|(index, (_, description))| {
            format!("{{\"letter\": {}, \"description\": {}}}",
                serde_json::to_string(&LETTERS.chars().nth(index).unwrap().to_string()).unwrap(),
                serde_json::to_string(description).unwrap())
        }).collect::<Vec<_>>().join(", ");
        // Preserve Python's insertion order and json.dumps separators: the
        // rendered prompt digest is an externally visible scoring field.
        let payload = format!("{{\"evidence\": {evidence}, \"criterion\": {criterion}, \"options\": [{options}]}}");
        json!([
            {"role":"system", "content":DIRECT_SYSTEM},
            {"role":"user", "content":payload}
        ])
    }
}

pub fn parse_judgment(body: &Value, model: &str) -> Result<JudgmentRow, ApiError> {
    let obj = body.as_object().ok_or_else(|| ApiError::new(400, "request body must be an object", "invalid_request_error"))?;
    let allowed = ["id", "state", "question", "options", "model", "timeout", "priority", "cache_salt"];
    let unknown: Vec<_> = obj.keys().filter(|key| !allowed.contains(&key.as_str())).cloned().collect();
    if !unknown.is_empty() { return Err(ApiError::new(400, format!("unexpected field(s): {}", unknown.join(", ")), "invalid_request_error")); }
    if let Some(value) = obj.get("model").filter(|value| !value.is_null()) {
        let got = value.as_str().filter(|value| !value.is_empty()).ok_or_else(|| ApiError::new(400, "model must be a nonempty string", "invalid_request_error"))?;
        if got != model { return Err(ApiError::new(404, format!("model '{got}' not found"), "model_not_found")); }
    }
    let id = obj.get("id").and_then(Value::as_str).filter(|value| !value.is_empty()).ok_or_else(|| ApiError::new(400, "id must be a nonempty string", "invalid_request_error"))?.to_owned();
    let question = obj.get("question").and_then(Value::as_str).filter(|value| !value.is_empty()).ok_or_else(|| ApiError::new(400, "question must be a nonempty string", "invalid_request_error"))?.to_owned();
    let state = obj.get("state").filter(|v| matches!(v, Value::String(s) if !s.is_empty()) || matches!(v, Value::Object(m) if !m.is_empty()) || matches!(v, Value::Array(a) if !a.is_empty())).cloned().ok_or_else(|| ApiError::new(400, "state must be a nonempty string, object, or array", "invalid_request_error"))?;
    let raw = obj.get("options").and_then(Value::as_array).filter(|items| (2..=16).contains(&items.len())).ok_or_else(|| ApiError::new(400, "options must contain 2-16 entries", "invalid_request_error"))?;
    let mut options = Vec::with_capacity(raw.len());
    for option in raw {
        let id = option.get("id").and_then(Value::as_str).ok_or_else(|| ApiError::new(400, "each option needs string id and description fields", "invalid_request_error"))?;
        let description = option.get("description").and_then(Value::as_str).ok_or_else(|| ApiError::new(400, "each option needs string id and description fields", "invalid_request_error"))?;
        options.push((id.to_owned(), description.to_owned()));
    }
    let ids: std::collections::HashSet<_> = options.iter().map(|(id, _)| id).collect();
    if ids.len() != options.len() { return Err(ApiError::new(400, "option IDs must be unique", "invalid_request_error")); }
    Ok(JudgmentRow { id, question, state, options })
}

/// Resolve answer slot IDs and prove each is one exact token whose
/// concatenation preserves the prompt's existing tokenization boundary.
pub fn answer_slots(pipeline: &Pipeline, prepared: &Prepared, labels: &[String]) -> Result<Vec<u32>, ApiError> {
    let (head, tail) = pipeline.prompt_tokenizer.split(&prepared.rendered);
    let tail_ids = pipeline.tokenizer.encode(&tail, false).map_err(tokenizer_error)?;
    let mut slots = Vec::with_capacity(labels.len());
    for label in labels {
        let ids = pipeline.tokenizer.encode(label, false).map_err(tokenizer_error)?;
        if ids.len() != 1 || pipeline.tokenizer.decode(&ids, false).ok().as_deref() != Some(label.as_str()) {
            return Err(ApiError::new(500, format!("served tokenizer cannot represent answer slot '{label}' as one exact token"), "scoring_unsupported"));
        }
        let mut expected = tail_ids.clone();
        expected.extend_from_slice(&ids);
        let combined = pipeline.tokenizer.encode(&format!("{tail}{label}"), false).map_err(tokenizer_error)?;
        if combined != expected {
            return Err(ApiError::new(500, format!("answer slot '{label}' changes tokenization at the prompt boundary"), "scoring_unsupported"));
        }
        slots.push(ids[0]);
    }
    if head.len() + tail_ids.len() != prepared.prompt_tokens.len() || labels.len() < 2 || labels.len() > 255 || slots.iter().copied().collect::<std::collections::HashSet<_>>().len() != slots.len() {
        return Err(ApiError::new(500, "scoring prompt or answer slots are unsupported", "scoring_unsupported"));
    }
    Ok(slots)
}

pub fn tokenizer_slot_labels(pipeline: &Pipeline) -> Vec<String> {
    let mut labels = Vec::with_capacity(255);
    for length in 1..=3 {
        let max = 26usize.pow(length as u32);
        for number in 0..max {
            let mut n = number;
            let mut chars = vec!['A'; length];
            for pos in (0..length).rev() {
                chars[pos] = (b'A' + (n % 26) as u8) as char;
                n /= 26;
            }
            let label: String = chars.into_iter().collect();
            if let Ok(ids) = pipeline.tokenizer.encode(&label, false) {
                if ids.len() == 1 && pipeline.tokenizer.decode(&ids, false).ok().as_deref() == Some(label.as_str()) {
                    labels.push(label);
                    if labels.len() == 255 { return labels; }
                }
            }
        }
    }
    labels
}

/// Mirrors Python's scoring prompt post-processing for chat formats that
/// expose a reasoning-start token: scoring runs with reasoning disabled,
/// and the prompt ends with an empty think block before the answer slot.
pub fn finalize_prompt(pipeline: &Pipeline, prepared: &mut Prepared) {
    if pipeline.chat_format.think_start_id.is_some() {
        prepared.rendered.push_str(crate::chat_format::THINK_START);
        prepared.rendered.push_str(crate::chat_format::THINK_END);
        prepared.prompt_tokens = pipeline.prompt_tokenizer.encode(&prepared.rendered);
    }
}

pub fn systemone_messages(state: &Value, spec: &SystemOneSpec, slots: &[String]) -> Value {
    let mut payload = format!("{{\"evidence\": {}", serde_json::to_string(state).unwrap());
    if let Some(instructions) = &spec.instructions {
        payload.push_str(&format!(", \"criterion\": {}", serde_json::to_string(instructions).unwrap()));
    }
    let options = slots.iter().zip(&spec.labels).zip(&spec.descriptions).map(|((slot,label),description)| {
        format!("{{\"slot\": {}, \"label\": {}, \"description\": {}}}",
            serde_json::to_string(slot).unwrap(), serde_json::to_string(label).unwrap(), serde_json::to_string(description).unwrap())
    }).collect::<Vec<_>>().join(", ");
    payload.push_str(&format!(", \"options\": [{options}]}}"));
    json!([
        {"role":"system", "content":SYSTEMONE_SYSTEM},
        {"role":"user", "content":payload}
    ])
}

pub fn systemone_answer(spec: &SystemOneSpec, probabilities: &[f64]) -> Value {
    if spec.kind == "noul" { return json!({"type":"noul", "noul":probabilities[0]}); }
    let confidence = concentration(probabilities);
    if spec.kind == "choice" {
        let best = (0..probabilities.len()).max_by(|a,b| probabilities[*a].total_cmp(&probabilities[*b])).unwrap_or(0);
        let dist: Map<String, Value> = spec.labels.iter().cloned().zip(probabilities.iter().copied().map(|p| json!(p))).collect();
        return json!({"type":"choice", "choice":spec.labels[best], "probabilities":dist, "confidence":confidence});
    }
    let score = probabilities.iter().enumerate().map(|(i,p)| i as f64 * p).sum::<f64>();
    let dist: Map<String, Value> = probabilities.iter().enumerate().map(|(i,p)| (i.to_string(), json!(p))).collect();
    json!({"type":"score", "score":score, "legend":spec.legend, "probabilities":dist, "confidence":confidence})
}

pub fn parse_systemone(body: &Value, model: &str) -> Result<(Value, Vec<SystemOneEntry>), ApiError> {
    let Some(obj) = body.as_object() else { return Err(systemone_validation(vec![field_error(&[], "request body must be an object", "model_type")])); };
    let mut errors = Vec::new();
    match obj.get("model").and_then(Value::as_str) {
        None => errors.push(field_error(&["model"], "field required", "missing")),
        Some(value) if value.is_empty() => errors.push(field_error(&["model"], "model must be a nonempty string", "value_error")),
        Some(value) if value != model => errors.push(field_error(&["model"], format!("model {value} is not served by this endpoint"), "value_error")),
        _ => {}
    }
    let state = match obj.get("state") {
        None => { errors.push(field_error(&["state"], "field required", "missing")); Value::Null },
        Some(value @ (Value::String(_) | Value::Object(_) | Value::Array(_))) => value.clone(),
        Some(_) => { errors.push(field_error(&["state"], "state must be a string, object, or array", "value_error")); Value::Null }
    };
    let questions = match obj.get("questions").and_then(Value::as_object) {
        None => { errors.push(field_error(&["questions"], "questions must be a nonempty object", "value_error")); None },
        Some(q) if q.is_empty() => { errors.push(field_error(&["questions"], "questions must be a nonempty object", "value_error")); None },
        Some(q) if q.len() > MAX_SYSTEMONE_QUESTIONS => { errors.push(field_error(&["questions"], format!("questions must contain at most {MAX_SYSTEMONE_QUESTIONS} entries"), "value_error")); None },
        Some(q) => Some(q),
    };
    let mut entries = Vec::new();
    if let Some(questions) = questions {
        for (qid, question) in questions {
            let loc = [&"questions"[..], qid.as_str()];
            let Some(q) = question.as_object() else { errors.push(field_error(&loc, "question must be an object", "model_type")); continue; };
            let kind = q.get("type").and_then(Value::as_str).unwrap_or("");
            if !["noul", "choice", "score"].contains(&kind) {
                errors.push(field_error(&["questions", qid, "type"], "type must be noul, choice, or score", "value_error")); continue;
            }
            let instructions = match q.get("instructions") {
                None | Some(Value::Null) => None,
                Some(value @ (Value::String(_) | Value::Object(_) | Value::Array(_))) => Some(value.clone()),
                Some(_) => { errors.push(field_error(&["questions", qid, "instructions"], "instructions must be a string, object, or array", "value_error")); None }
            };
            let criteria = q.get("criteria");
            let (labels, descriptions, legend, deterministic) = match kind {
                "noul" => {
                    if let Some(value) = criteria.filter(|v| !v.is_null()) {
                        let Some(map) = value.as_object() else { errors.push(field_error(&["questions", qid, "criteria"], "noul criteria must map true/false to descriptions", "value_error")); continue; };
                        if ["true", "false"].iter().any(|key| map.get(*key).is_some_and(|v| !description_value(v, false))) { errors.push(field_error(&["questions", qid, "criteria"], "noul criteria descriptions must be strings, objects, arrays, or null", "value_error")); }
                    }
                    (vec!["true".into(), "false".into()], ["true", "false"].iter().map(|key| criteria.and_then(Value::as_object).and_then(|m| m.get(*key)).filter(|v| !v.is_null()).cloned().unwrap_or(json!(key))).collect(), None, false)
                }
                "choice" => {
                    let Some(map) = criteria.and_then(Value::as_object).filter(|m| !m.is_empty()) else { errors.push(field_error(&["questions", qid, "criteria"], "choice criteria must be a nonempty object mapping labels to descriptions", "value_error")); continue; };
                    if map.len() > 255 { errors.push(field_error(&["questions", qid, "criteria"], "choice supports at most 255 options", "value_error")); continue; }
                    let invalid = map.iter().any(|(k,v)| k.is_empty() || !description_value(v, true));
                    if invalid { errors.push(field_error(&["questions", qid, "criteria"], "choice labels must be strings and descriptions strings, objects, arrays, or null", "value_error")); continue; }
                    let labels = map.keys().cloned().collect::<Vec<_>>();
                    let descriptions = map.iter().map(|(label,v)| if v.is_null() { json!(label) } else {v.clone()}).collect();
                    let deterministic = labels.len()==1;
                    (labels, descriptions, None, deterministic)
                }
                _ => {
                    let Some(values) = criteria.and_then(Value::as_array).filter(|v| !v.is_empty()) else { errors.push(field_error(&["questions", qid, "criteria"], "score criteria must be a nonempty array of level descriptions", "value_error")); continue; };
                    if values.len() > 255 { errors.push(field_error(&["questions", qid, "criteria"], "score supports at most 255 levels", "value_error")); continue; }
                    if values.iter().any(|v| !description_value(v, false)) { errors.push(field_error(&["questions", qid, "criteria"], "score descriptions must be strings, objects, or arrays", "value_error")); continue; }
                    let labels = (0..values.len()).map(|i| i.to_string()).collect::<Vec<_>>();
                    let legend = json!(labels.iter().zip(values).map(|(k,v)| (k.clone(),v.clone())).collect::<Map<_,_>>());
                    (labels, values.clone(), Some(legend), values.len()==1)
                }
            };
            entries.push(SystemOneEntry { id: qid.clone(), spec: SystemOneSpec { kind: kind.into(), instructions, labels, descriptions, legend, deterministic } });
        }
    }
    if !errors.is_empty() { return Err(systemone_validation(errors)); }
    Ok((state, entries))
}

fn description_value(value: &Value, allow_null: bool) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_)) || (allow_null && value.is_null())
}

pub fn run_score(pipeline: &Pipeline, prepared: &Prepared, slots: Vec<u32>, deadline: Option<std::time::Duration>, options: GenerationFrameOptions, disconnect_stream: Option<&std::net::TcpStream>) -> Result<(crate::protocol::DoneEvent, Vec<f64>, Vec<u32>), PipelineError> {
    let mut generation = pipeline.generation_request_with_frame_options(
        prepared, 0, None, deadline, SamplingParameters::default(), options,
    );
    generation.frame.score_tokens = slots.clone();
    let call = pipeline.client.submit(&generation, None, None)?;
    let _disconnect_watch = match disconnect_stream {
        Some(stream) => Some(crate::serve::server::DisconnectWatch::start(stream, pipeline.client.clone(), vec![call.clone()]).map_err(|error| {
            pipeline.client.cancel(&call);
            PipelineError { status: 500, message: format!("could not monitor request disconnect: {error}"), code: Some("internal_error".into()) }
        })?),
        None => None,
    };
    let done = call.result(deadline)?;
    let logits = done.option_logits.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
    let probabilities = softmax(&logits).map_err(|message| PipelineError { status: 500, message, code: Some("scoring_error".into()) })?;
    Ok((done, probabilities, slots))
}

pub fn scoring_frame_options(body: &Value, pipeline: &Pipeline, prepared: &Prepared) -> Result<GenerationFrameOptions, PipelineError> {
    let mut controls = Map::new();
    for key in ["priority", "cache_salt"] {
        if let Some(value) = body.get(key) { controls.insert(key.to_owned(), value.clone()); }
    }
    let mut options = crate::serve::pipeline::generation_frame_options(&Value::Object(controls), prepared, &pipeline.chat_format, None)?;
    // Direct scoring always has the same neutral score-only job settings.
    options.seed = 0;
    options.flags = 0;
    options.bias_tokens.clear();
    options.bias_values.clear();
    options.thinking_budget = None;
    options.logprobs = false;
    options.top_logprobs = 0;
    Ok(options)
}

pub fn softmax(values: &[f64]) -> Result<Vec<f64>, String> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) { return Err("need at least two finite scores".into()); }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 { return Err("score distribution is not finite".into()); }
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

pub fn concentration(probabilities: &[f64]) -> f64 {
    if probabilities.len() < 2 { return 1.0; }
    let entropy = probabilities.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum::<f64>();
    (1.0 - entropy / (probabilities.len() as f64).ln()).max(0.0)
}

pub fn judgment_answer(model: &str, row: &JudgmentRow, prompt: &str, done: &crate::protocol::DoneEvent, probabilities: &[f64], slot_ids: &[u32]) -> Value {
    json!({
        "id": row.id, "option_ids": row.options.iter().map(|(id, _)| id).collect::<Vec<_>>(),
        "probabilities": probabilities, "option_logits": done.option_logits,
        "input_tokens": done.prompt_tokens, "answer_token_ids": slot_ids,
        "prompt_sha256": crate::support::sha256::hex_digest(prompt.as_bytes()),
        "prompt_version": PROMPT_VERSION, "model": {"id": model}, "readout": READOUT,
        "probability_status": PROBABILITY_STATUS,
        "forward_seconds": done.prefill_micros as f64 / 1_000_000.0,
        "total_seconds": done.wall_micros as f64 / 1_000_000.0,
        "usage": {"prompt_tokens": done.prompt_tokens, "completion_tokens": 0, "total_tokens": done.prompt_tokens}
    })
}

pub fn systemone_validation(details: Vec<crate::support::errors::FieldError>) -> ApiError {
    crate::support::errors::RequestValidationError::new(details).into_api_error()
}

pub fn unsupported_systemone(path: &[&str], message: &str) -> ApiError {
    systemone_validation(vec![field_error(path, message, "value_error")])
}

fn tokenizer_error(error: crate::tokenization::fast::TokenizerError) -> ApiError {
    ApiError::new(500, error.0, "internal_error")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judgments_validate_rows_and_preserve_python_payload_order() {
        let row = parse_judgment(&json!({
            "id":"r1", "state":{"x":1}, "question":"which?",
            "options":[{"id":"yes","description":"Yes"},{"id":"no","description":"No"}]
        }), "model").unwrap();
        let messages = row.messages();
        assert_eq!(messages[0]["content"], DIRECT_SYSTEM);
        assert_eq!(messages[1]["content"], "{\"evidence\": {\"x\":1}, \"criterion\": \"which?\", \"options\": [{\"letter\": \"A\", \"description\": \"Yes\"}, {\"letter\": \"B\", \"description\": \"No\"}]}");
        assert!(parse_judgment(&json!({"id":"r1","state":{},"question":"q","options":[]}), "model").is_err());
        assert_eq!(parse_judgment(&json!({"id":"r1","state":"s","question":"q","options":[{"id":"a","description":"a"},{"id":"a","description":"b"}]}), "model").unwrap_err().code, "invalid_request_error");
    }

    #[test]
    fn softmax_is_stable_and_rejects_invalid_scores() {
        let p = softmax(&[1000.0, 1001.0]).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[1] > p[0]);
        assert!(softmax(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn systemone_parses_all_question_types_and_resolves_slot_semantics() {
        let (_, entries) = parse_systemone(&json!({
            "model":"m", "state":"evidence", "questions":{
                "n":{"type":"noul"},
                "q":{"type":"choice","criteria":{"yes":"ok","no":"bad"}},
                "s":{"type":"score","criteria":["low","high"]}
            }
        }), "m").unwrap();
        assert_eq!(entries.len(), 3);
        assert!(entries.iter().any(|entry| entry.spec.kind == "noul"));
        assert!(entries.iter().find(|entry| entry.spec.kind == "score").unwrap().spec.legend.is_some());
        let error = parse_systemone(&json!({"model":"m","state":"evidence","questions":{"bad":{"type":"choice","criteria":{}}}}), "m").unwrap_err();
        assert!(error.validation_details.is_some());
    }
}
