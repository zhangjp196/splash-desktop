//! The tool calls a model writes, parsed out of the output as it arrives
//! into OpenAI JSON argument deltas, for streamed and complete responses
//! alike. Ported from `server/output.py`'s `StreamingToolCallProjector`.
//!
//! The parameter types a call's values convert by come from a
//! [`ParameterTypes`] provider: the full schema reader is tool_schema's
//! (module 11); the tests supply simple providers.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::chat_format::{ChatFormat, THINK_END, TOOL_CALL_CLOSE};
use crate::output::stop::hold_partial;
use crate::support::json_codec;

/// The longest stop sequence a request may carry.
pub const MAX_STOP_CHARACTERS: usize = 16 * 1024;
/// The longest a call's or a parameter's name may be.
pub const MAX_NAME_LENGTH: usize = 256;
/// The space a name loses around it.
pub const NAME_SPACE: &str = " \t\n\r";
/// Keep individual SSE frames bounded even when a tool has a large string
/// argument. Callers preserve fragment order.
pub const TOOL_ARGUMENT_DELTA_CHARS: usize = 16 * 1024;
/// The deepest a call's arguments nest as an object the server writes.
pub const MAX_ARGUMENTS_DEPTH: usize = 128 - 5;

/// The JSON types a parameter's value converts by; empty means the value
/// is its text and streams as it is written.
pub type Types = HashSet<String>;

/// The types of the tools a request's calls name, read as they name them.
pub trait ParameterTypes: Send + Sync {
    /// The types of tool `name`'s parameters, and those of any other.
    fn of(&self, name: &str) -> (HashMap<String, Types>, Option<Types>);
}

/// The types provider of a request without a tool policy: every value is
/// its text and streams as it is written.
pub struct NoTypes;

impl ParameterTypes for NoTypes {
    fn of(&self, _name: &str) -> (HashMap<String, Types>, Option<Types>) {
        (HashMap::new(), None)
    }
}

/// A provider from a static map, as the tests use.
pub struct StaticTypes {
    pub tools: BTreeMap<String, (HashMap<String, Types>, Option<Types>)>,
}

impl ParameterTypes for StaticTypes {
    fn of(&self, name: &str) -> (HashMap<String, Types>, Option<Types>) {
        self.tools
            .get(name)
            .cloned()
            .unwrap_or((HashMap::new(), None))
    }
}

/// The JSON value `text` spells, if it can be written back as JSON in its
/// call's arguments: its numbers finite, its strings Unicode and its
/// containers nested at most MAX_ARGUMENTS_DEPTH deep. Otherwise `None`.
fn json_value(text: &str) -> Option<serde_json::Value> {
    let value = json_codec::loads(text).ok()?;
    let mut pending = vec![(value.clone(), 2usize)];
    while let Some((item, depth)) = pending.pop() {
        match &item {
            serde_json::Value::String(text) => {
                if text.chars().any(|c| (0xD800..0xE000).contains(&(c as u32))) {
                    return None;
                }
            }
            serde_json::Value::Array(items) => {
                if depth > MAX_ARGUMENTS_DEPTH {
                    return None;
                }
                pending.extend(items.iter().map(|child| (child.clone(), depth + 1)));
            }
            serde_json::Value::Object(map) => {
                if depth > MAX_ARGUMENTS_DEPTH {
                    return None;
                }
                pending.extend(map.values().map(|child| (child.clone(), depth + 1)));
            }
            _ => {}
        }
    }
    Some(value)
}

/// A parameter's text as the JSON types its schema gives for it read it;
/// `types` empty means the text as it is. A string reads as the text, and
/// another value as the JSON the text spells when that is of a declared
/// type or when no declared type is a string. A boolean or null declared
/// without a string may also be spelled as Python spells it. Otherwise the
/// text.
pub fn convert_value(text: &str, types: &Types) -> serde_json::Value {
    let json_type = |value: &serde_json::Value| match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    };
    match json_value(text) {
        Some(value) => {
            if types.contains("string") {
                let kind = json_type(&value);
                let declared = types.contains(kind)
                    || (kind == "integer" && types.contains("number"));
                if declared && kind != "string" {
                    return value;
                }
                serde_json::Value::String(text.to_string())
            } else {
                value
            }
        }
        None => {
            let literal = match text.trim() {
                "True" | "true" => Some(serde_json::Value::Bool(true)),
                "False" | "false" => Some(serde_json::Value::Bool(false)),
                "None" | "null" => Some(serde_json::Value::Null),
                _ => None,
            };
            match literal {
                Some(value) if !types.contains("string") && types.contains(json_type(&value)) => {
                    value
                }
                _ => serde_json::Value::String(text.to_string()),
            }
        }
    }
}

/// One event the projector emits as the output arrives.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectorEvent {
    /// Text outside calls.
    Content(String),
    /// A call opened: its index, its id and its function's name.
    ToolCall {
        index: usize,
        id: String,
        name: String,
    },
    /// A piece of a call's arguments JSON.
    ToolDelta { index: usize, chunk: String },
}

/// One finished call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// The projector's state.
struct ProjectorState {
    tool_types: Box<dyn ParameterTypes>,
    request_id: u64,
    syntax: ChatFormat,
    /// The tags that end the text outside calls: a call's opening where a
    /// policy stands, and the reasoning's close.
    text_tags: Vec<String>,
    pending: String,
    state: State,
    call_index: usize,
    /// The open call, `None` while the text names none or names no
    /// function.
    call_id: Option<String>,
    function_name: Option<String>,
    parameter_types: HashMap<String, Types>,
    other_types: Option<Types>,
    parameter_names: HashSet<String>,
    argument_fragments: Vec<String>,
    parameter_count: usize,
    /// The open parameter, `None` while its value is dropped.
    parameter_name: Option<String>,
    value_types: Option<Types>,
    value_streams: bool,
    value_started: bool,
    value_parts: Vec<String>,
    content_fragments: Vec<String>,
    /// How many content fragments the stream has published (the rest are
    /// whitespace it holds), and whether the text since the start of the
    /// output or the last call has shown a visible character yet.
    streamed_count: usize,
    text_visible: bool,
    closed_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Output,
    Content,
    Json,
    Name,
    Arguments,
    Parameter,
    Value,
    Closing,
}

/// The projector of one request's output.
pub struct StreamingToolCallProjector {
    state: ProjectorState,
}

impl StreamingToolCallProjector {
    /// The projector of one request: its tool policy's types, its id, and
    /// the layout its calls are read in, which a tool policy needs.
    pub fn new(
        tool_types: Box<dyn ParameterTypes>,
        request_id: u64,
        syntax: &ChatFormat,
        structured: bool,
    ) -> StreamingToolCallProjector {
        let has_policy = true;
        let text_tags = if has_policy {
            vec![syntax.call_syntax.call_open(), THINK_END.to_string()]
        } else {
            vec![THINK_END.to_string()]
        };
        StreamingToolCallProjector {
            state: ProjectorState {
                tool_types,
                request_id,
                syntax: syntax.clone(),
                text_tags,
                pending: String::new(),
                state: if structured { State::Output } else { State::Content },
                call_index: 0,
                call_id: None,
                function_name: None,
                parameter_types: HashMap::new(),
                other_types: None,
                parameter_names: HashSet::new(),
                argument_fragments: Vec::new(),
                parameter_count: 0,
                parameter_name: None,
                value_types: None,
                value_streams: false,
                value_started: false,
                value_parts: Vec::new(),
                content_fragments: Vec::new(),
                streamed_count: 0,
                text_visible: false,
                closed_calls: Vec::new(),
            },
        }
    }

    /// The projector of an output without a tool policy: no calls, the
    /// openings are text, and only the reasoning's closes are dropped.
    pub fn without_policy(request_id: u64, syntax: &ChatFormat) -> StreamingToolCallProjector {
        let mut projector = Self::new(Box::new(NoTypes), request_id, syntax, false);
        projector.state.text_tags = vec![THINK_END.to_string()];
        projector.state.text_visible = true;
        projector
    }

    /// Whether the output so far ends inside a call, past its opening.
    pub fn in_call(&self) -> bool {
        matches!(
            self.state.state,
            State::Name | State::Arguments | State::Parameter | State::Value
        )
    }

    fn emit_content(&mut self, value: &str, events: &mut Vec<ProjectorEvent>) {
        if value.is_empty() {
            return;
        }
        self.state.content_fragments.push(value.to_string());
        // The chat template sets calls apart from text with whitespace.
        // Hold whitespace that starts the output or follows a call until
        // visible text arrives. Before the first text or after the last,
        // it only frames the calls and is dropped; between two texts it
        // separates them and streams with the later one.
        if !self.state.text_visible {
            if value.trim().is_empty() {
                return;
            }
            self.state.text_visible = true;
        }
        let streamed = self.state.streamed_count;
        let unsent: String = self.state.content_fragments[streamed..].concat();
        self.state.streamed_count = self.state.content_fragments.len();
        if !unsent.is_empty() {
            events.push(ProjectorEvent::Content(unsent));
        }
    }

    fn streamed_content(&self) -> String {
        self.state.content_fragments[..self.state.streamed_count].concat()
    }

    fn emit_argument(&mut self, fragment: &str, events: &mut Vec<ProjectorEvent>) {
        if self.state.call_id.is_none() {
            return;
        }
        self.state.argument_fragments.push(fragment.to_string());
        for offset in (0..fragment.len()).step_by(TOOL_ARGUMENT_DELTA_CHARS) {
            let end = (offset + TOOL_ARGUMENT_DELTA_CHARS).min(fragment.len());
            // A char boundary: the fragments are JSON text, ASCII except
            // string values, so step by chars.
            let mut end = end;
            while end < fragment.len() && !fragment.is_char_boundary(end) {
                end += 1;
            }
            events.push(ProjectorEvent::ToolDelta {
                index: self.state.call_index,
                chunk: fragment[offset..end].to_string(),
            });
        }
    }

    fn begin_call(&mut self, name: &str, events: &mut Vec<ProjectorEvent>) {
        if name.trim_matches(|c| NAME_SPACE.contains(c)).is_empty() {
            // Nothing of a call has streamed yet: the opening, and the text
            // after it, are content.
            self.state.state = State::Content;
            let opening = format!("{}{}", self.state.syntax.call_syntax.call_open(), name);
            self.emit_content(&opening, events);
            return;
        }
        self.state.state = State::Arguments;
        self.state.argument_fragments.clear();
        self.state.parameter_count = 0;
        self.state.parameter_names.clear();
        let name = name.trim_matches(|c| NAME_SPACE.contains(c)).to_string();
        if self.state.streamed_count == 0 {
            // Whitespace before the first text only framed the calls.
            self.state.content_fragments.clear();
        }
        self.state.function_name = Some(name.clone());
        let call_id = format!("call_{}_{}", self.state.request_id, self.state.call_index);
        self.state.call_id = Some(call_id.clone());
        let (parameter_types, other_types) = self.state.tool_types.of(&name);
        self.state.parameter_types = parameter_types;
        self.state.other_types = other_types;
        events.push(ProjectorEvent::ToolCall {
            index: self.state.call_index,
            id: call_id,
            name,
        });
        self.emit_argument("{", events);
    }

    fn key(&mut self, name: &str) -> String {
        let separator = if self.state.parameter_count != 0 { "," } else { "" };
        self.state.parameter_count += 1;
        format!(
            "{separator}{}:",
            json_codec::dumps(&serde_json::Value::String(name.to_string()))
                .unwrap_or_else(|_| format!("\"{name}\""))
        )
    }

    fn begin_parameter(&mut self, name: &str, events: &mut Vec<ProjectorEvent>) {
        let name = name.trim_matches(|c| NAME_SPACE.contains(c)).to_string();
        // A repeated parameter keeps its first value, which may have
        // streamed; a parameter without a name, or of a call that names no
        // function, keeps none.
        let kept = !name.is_empty()
            && !self.state.parameter_names.contains(&name)
            && self.state.call_id.is_some();
        self.state.parameter_names.insert(name.clone());
        self.state.parameter_name = kept.then(|| name.clone());
        self.state.value_types = self
            .state
            .parameter_types
            .get(&name)
            .cloned()
            .or_else(|| self.state.other_types.clone());
        // A value whose schema gives it no type, or the type string alone,
        // is its text and streams as it is written; any other waits for
        // its end to convert.
        self.state.value_streams = self
            .state
            .value_types
            .as_ref()
            .map(|types| types.is_empty() || (types.len() == 1 && types.contains("string")))
            .unwrap_or(true);
        self.state.value_started = false;
        self.state.value_parts.clear();
        if self.state.value_streams && self.state.parameter_name.is_some() {
            let key = self.key(&name);
            self.emit_argument(&format!("{key}\""), events);
        }
        self.state.state = State::Value;
    }

    fn put_value(&mut self, mut text: &str, events: &mut Vec<ProjectorEvent>) {
        if self.state.parameter_name.is_none() {
            return;
        }
        if !self.state.value_started {
            if text.is_empty() {
                return;
            }
            self.state.value_started = true;
            let open = self.state.syntax.call_syntax.value_open;
            text = text.strip_prefix(open).unwrap_or(text);
        }
        if text.is_empty() {
            return;
        }
        if self.state.value_streams {
            let escaped = json_codec::dumps(&serde_json::Value::String(text.to_string()))
                .unwrap_or_default();
            self.emit_argument(&escaped[1..escaped.len() - 1], events);
        } else {
            self.state.value_parts.push(text.to_string());
        }
    }

    fn end_parameter(&mut self, text: &str, events: &mut Vec<ProjectorEvent>) {
        if self.state.parameter_name.is_some() {
            let mut text = text.to_string();
            if !self.state.value_started {
                self.state.value_started = true;
                let open = self.state.syntax.call_syntax.value_open;
                text = text.strip_prefix(open).unwrap_or(&text).to_string();
            }
            self.put_value(&text, events);
            if self.state.value_streams {
                self.emit_argument("\"", events);
            } else {
                let value = convert_value(
                    &self.state.value_parts.concat(),
                    self.state.value_types.as_ref().unwrap_or(&Types::new()),
                );
                let name = self.state.parameter_name.clone().expect("checked above");
                let key = self.key(&name);
                let rendered = json_codec::dumps(&value)
                    .unwrap_or_else(|_| format!("\"{value:?}\""));
                self.emit_argument(&format!("{key}{rendered}"), events);
            }
        }
        self.state.parameter_name = None;
        self.state.value_parts.clear();
        self.state.state = State::Arguments;
    }

    fn finish_call(&mut self, events: &mut Vec<ProjectorEvent>) {
        self.emit_argument("}", events);
        if let Some(call_id) = self.state.call_id.clone() {
            self.state.closed_calls.push(ToolCall {
                id: call_id,
                name: self.state.function_name.clone().unwrap_or_default(),
                arguments: self.state.argument_fragments.concat(),
            });
            self.state.call_index += 1;
        }
        self.state.call_id = None;
        self.state.function_name = None;
        self.state.argument_fragments.clear();
        self.state.text_visible = false;
        self.state.state = State::Closing;
    }

    /// The tag of `tags` that the pending text spells first, and where.
    fn next_tag(&self, tags: &[String]) -> (usize, Option<String>) {
        let mut found: Vec<(usize, &String)> = tags
            .iter()
            .filter_map(|tag| self.state.pending.find(tag).map(|index| (index, tag)))
            .collect();
        found.sort_by_key(|(index, _)| *index);
        found
            .into_iter()
            .next()
            .map(|(index, tag)| (index, Some(tag.clone())))
            .unwrap_or((usize::MAX, None))
    }

    /// The name the pending text spells up to `end`, taken off the text
    /// with `end` unless `end` is kept; "" for a longer one, and `None`
    /// while it may yet end.
    fn take_name(&mut self, end: &str, keep: bool) -> Option<String> {
        let limit = MAX_NAME_LENGTH + end.len();
        let index = self.state.pending.find(end).filter(|index| *index <= limit);
        match index {
            None => {
                if self.state.pending.len() < limit {
                    None
                } else {
                    Some(String::new())
                }
            }
            Some(index) => {
                let name = self.state.pending[..index].to_string();
                self.state.pending = if keep {
                    self.state.pending[index..].to_string()
                } else {
                    self.state.pending[index + end.len()..].to_string()
                };
                Some(name)
            }
        }
    }

    /// The output as it arrives, split into its events.
    pub fn put(&mut self, text: &str) -> Vec<ProjectorEvent> {
        self.state.pending.push_str(text);
        let mut events = Vec::new();
        loop {
            match self.state.state {
                State::Output => {
                    let first = self.state.pending.trim_start();
                    if first.is_empty() {
                        break;
                    }
                    // A structured answer is one JSON value or tool calls.
                    // Once JSON starts, XML spellings inside its strings
                    // are just data.
                    self.state.state = if first.starts_with('<') {
                        State::Content
                    } else {
                        State::Json
                    };
                }
                State::Json => {
                    let pending = std::mem::take(&mut self.state.pending);
                    self.emit_content(&pending, &mut events);
                    break;
                }
                State::Content => {
                    let (start, tag) = self.next_tag(&self.state.text_tags);
                    let Some(tag) = tag else {
                        let markers: Vec<&str> =
                            self.state.text_tags.iter().map(|t| t.as_str()).collect();
                        let pending = self.state.pending.clone();
                        let (ready, held) = hold_partial(&pending, &markers);
                        self.state.pending = held.to_string();
                        self.emit_content(ready, &mut events);
                        break;
                    };
                    let ready = self.state.pending[..start].to_string();
                    self.emit_content(&ready, &mut events);
                    self.state.pending =
                        self.state.pending[start + tag.len()..].to_string();
                    if tag != THINK_END {
                        // A call's opening, the other tag.
                        self.state.state = State::Name;
                    }
                    continue;
                }
                State::Name => {
                    // The character that ends the name stays: the arguments
                    // drop it, or read the tag it begins.
                    let end = self.state.syntax.call_syntax.name_end().to_string();
                    let Some(name) = self.take_name(&end, true) else {
                        break;
                    };
                    self.begin_call(&name, &mut events);
                    continue;
                }
                State::Arguments => {
                    // Text between parameters is dropped, but for a tag it
                    // may begin.
                    let tags = self.state.syntax.call_syntax.argument_tags();
                    let (start, tag) = self.next_tag(&tags.iter().map(|t| t.to_string()).collect::<Vec<_>>());
                    let Some(tag) = tag else {
                        let pending = self.state.pending.clone();
                        let held = hold_partial(&pending, &tags).1;
                        self.state.pending = held.to_string();
                        break;
                    };
                    self.state.pending = self.state.pending[start + tag.len()..].to_string();
                    if tag == self.state.syntax.call_syntax.key_open {
                        self.state.state = State::Parameter;
                    } else {
                        self.finish_call(&mut events);
                        if tag == TOOL_CALL_CLOSE {
                            self.state.state = State::Content;
                        }
                    }
                    continue;
                }
                State::Parameter => {
                    let end = self.state.syntax.call_syntax.key_close.to_string();
                    let Some(name) = self.take_name(&end, false) else {
                        break;
                    };
                    self.begin_parameter(&name, &mut events);
                    continue;
                }
                State::Value => {
                    // Before the value starts, the start of the value_open
                    // the template sets before it waits for the rest.
                    let open = self.state.syntax.call_syntax.value_open;
                    if !self.state.value_started
                        && self.state.pending.len() < open.len()
                        && open.starts_with(&self.state.pending)
                    {
                        break;
                    }
                    let ends = self.state.syntax.call_syntax.value_ends();
                    let end_refs: Vec<&str> = ends.iter().map(|t| t.as_str()).collect();
                    let (start, tag) = self.next_tag(&end_refs.iter().map(|t| t.to_string()).collect::<Vec<_>>());
                    let Some(_tag) = tag else {
                        let pending = self.state.pending.clone();
                        let held = hold_partial(&pending, &end_refs).1;
                        let ready_len = pending.len() - held.len();
                        let ready = pending[..ready_len].to_string();
                        self.put_value(&ready, &mut events);
                        self.state.pending = held.to_string();
                        break;
                    };
                    let text = self.state.pending[..start].to_string();
                    self.end_parameter(&text, &mut events);
                    // The tag after the close stays for the arguments to
                    // read.
                    let close = self.state.syntax.call_syntax.value_close;
                    self.state.pending = self.state.pending[start + close.len()..].to_string();
                    continue;
                }
                State::Closing => {
                    // After the arguments' close, the template closes the
                    // call's block.
                    let rest = self
                        .state
                        .pending
                        .trim_start_matches(|c| NAME_SPACE.contains(c))
                        .to_string();
                    if rest.starts_with(TOOL_CALL_CLOSE) {
                        self.state.pending = rest[TOOL_CALL_CLOSE.len()..].to_string();
                    } else if TOOL_CALL_CLOSE.starts_with(&rest) {
                        break;
                    }
                    self.state.state = State::Content;
                }
            }
        }
        events
    }

    /// The content and calls of the output, and the events the stream
    /// still owes for what `put` held back.
    pub fn finish(
        &mut self,
        incomplete: bool,
    ) -> (String, Vec<ToolCall>, Vec<ProjectorEvent>) {
        let mut events = Vec::new();
        if !incomplete {
            if self.state.state == State::Name {
                let name = if self.state.pending.len() <= MAX_NAME_LENGTH {
                    self.state.pending.clone()
                } else {
                    String::new()
                };
                self.state.pending.clear();
                self.begin_call(&name, &mut events);
            }
            if self.state.state == State::Value {
                self.state.pending.clear();
                self.end_parameter("", &mut events);
            }
            if matches!(self.state.state, State::Arguments | State::Parameter) {
                self.state.pending.clear();
                self.finish_call(&mut events);
            }
        }
        if self.state.state == State::Closing {
            self.state.pending.clear();
            self.state.state = State::Content;
        }
        if matches!(
            self.state.state,
            State::Content | State::Output | State::Json
        ) && !self.state.pending.is_empty()
            && !(incomplete
                && self
                    .state
                    .text_tags
                    .iter()
                    .any(|tag| tag.starts_with(&self.state.pending)))
        {
            self.state.content_fragments.push(self.state.pending.clone());
        } else if !self.state.closed_calls.is_empty() {
            // Whitespace held after the last text only framed the calls.
            self.state
                .content_fragments
                .truncate(self.state.streamed_count);
        }
        self.state.pending.clear();
        let mut calls = self.state.closed_calls.clone();
        if let Some(call_id) = &self.state.call_id {
            calls.push(ToolCall {
                id: call_id.clone(),
                name: self.state.function_name.clone().unwrap_or_default(),
                arguments: self.state.argument_fragments.concat(),
            });
        }
        let streamed = self.streamed_content();
        let content = if !calls.is_empty() && incomplete {
            streamed.clone()
        } else {
            self.state.content_fragments.concat()
        };
        let unsent = &content[streamed.len()..];
        if !unsent.is_empty() {
            events.push(ProjectorEvent::Content(unsent.to_string()));
        }
        (content, calls, events)
    }
}
