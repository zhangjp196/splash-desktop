//! Prometheus text exposition for the Python server's native status schema.

use serde_json::Value;
use std::sync::Mutex;
use std::time::Duration;

const SERIES: &[(&str, &[&str])] = &[
    ("splash_ready", &["ready"]),
    ("splash_metal_healthy", &["metal", "healthy"]),
    ("splash_metal_pipelines_created_while_serving_total", &["metal", "pipelines_created_while_serving"]),
    ("splash_console_dropped_lines_total", &["console", "dropped_lines"]),
    ("splash_console_queued", &["console", "queued"]),
    ("splash_transport_pending", &["transport", "pending"]),
    ("splash_frontend_active", &["frontend", "active"]),
    ("splash_frontend_waiting", &["frontend", "waiting"]),
    ("splash_requests_submitted_total", &["requests", "submitted"]),
    ("splash_requests_completed_total", &["requests", "completed"]),
    ("splash_requests_cancelled_total", &["requests", "cancelled"]),
    ("splash_requests_failed_total", &["requests", "failed"]),
    ("splash_scheduler_queued", &["scheduler", "queued"]),
    ("splash_scheduler_waiting_resources", &["scheduler", "waiting_resources"]),
    ("splash_scheduler_waiting_prefix", &["scheduler", "waiting_prefix"]),
    ("splash_cache_resource_suspensions_total", &["cache", "resource_suspensions"]),
    ("splash_cache_priority_suspensions_total", &["cache", "priority_suspensions"]),
    ("splash_cache_resource_resumptions_total", &["cache", "resource_resumptions"]),
    ("splash_cache_resource_replay_tokens_total", &["cache", "resource_replay_tokens"]),
    ("splash_scheduler_prefilling", &["scheduler", "prefilling"]),
    ("splash_scheduler_decoding", &["scheduler", "decoding"]),
    ("splash_scheduler_waiting_mask", &["scheduler", "waiting_mask"]),
    ("splash_scheduler_prefill_batches_total", &["scheduler", "prefill_batches"]),
    ("splash_scheduler_prefill_rows_total", &["scheduler", "prefill_rows"]),
    ("splash_scheduler_decode_batches_total", &["scheduler", "decode_batches"]),
    ("splash_scheduler_decode_b1_total", &["scheduler", "decode_batches_by_width", "b1"]),
    ("splash_scheduler_decode_b2_total", &["scheduler", "decode_batches_by_width", "b2"]),
    ("splash_scheduler_decode_b3_total", &["scheduler", "decode_batches_by_width", "b3"]),
    ("splash_scheduler_decode_b4_total", &["scheduler", "decode_batches_by_width", "b4"]),
    ("splash_kv_pages_allocated", &["kv", "pages_allocated"]),
    ("splash_kv_pages_active", &["kv", "pages_active"]),
    ("splash_kv_pages_cache", &["kv", "pages_cache"]),
    ("splash_kv_free_allocated_pages", &["kv", "pages_free"]),
    ("splash_kv_allocated_bytes", &["kv", "allocated_bytes"]),
    ("splash_kv_reclaimable_bytes", &["kv", "reclaimable_bytes"]),
    ("splash_kv_extent_allocations_total", &["kv", "extent_allocations"]),
    ("splash_kv_extent_releases_total", &["kv", "extent_releases"]),
    ("splash_kv_extent_allocate_max_milliseconds", &["kv", "extent_allocate_max_ms"]),
    ("splash_kv_extent_release_max_milliseconds", &["kv", "extent_release_max_ms"]),
    ("splash_kv_extent_compactions_total", &["kv", "extent_compactions"]),
    ("splash_kv_pages_moved_total", &["kv", "pages_moved"]),
    ("splash_kv_extent_compact_max_milliseconds", &["kv", "extent_compact_max_ms"]),
    ("splash_state_entries", &["state", "entries"]),
    ("splash_state_pinned", &["state", "pinned"]),
    ("splash_state_in_use", &["state", "in_use"]),
    ("splash_state_in_use_evictions_total", &["state", "in_use_evictions"]),
    ("splash_state_bytes", &["state", "bytes"]),
    ("splash_state_active_lanes", &["state", "active_lanes"]),
    ("splash_state_publications_total", &["state", "publications"]),
    ("splash_state_evictions_total", &["state", "evictions"]),
    ("splash_disk_capacity_bytes", &["disk", "capacity_bytes"]),
    ("splash_disk_limit_bytes", &["disk", "limit_bytes"]),
    ("splash_disk_used_bytes", &["disk", "used_bytes"]),
    ("splash_disk_volume_available_bytes", &["disk", "volume_available_bytes"]),
    ("splash_disk_write_budget_bytes", &["disk", "write_budget_bytes"]),
    ("splash_disk_read_bytes_total", &["disk", "read_bytes"]),
    ("splash_disk_written_bytes_total", &["disk", "written_bytes"]),
    ("splash_cache_hits_total", &["cache", "hits"]),
    ("splash_cache_cold_misses_total", &["cache", "cold_misses"]),
    ("splash_cache_reused_tokens_total", &["cache", "reused_tokens"]),
    ("splash_cache_kv_disk_hit_rate", &["cache", "kv_disk_hit_rate"]),
    ("splash_cache_deduplicated_kv_tokens_total", &["cache", "deduplicated_kv_tokens"]),
    ("splash_cache_lazy_junctions_total", &["cache", "lazy_junctions"]),
    ("splash_target_prefill_rows_total", &["draft_context", "target_prefill_rows"]),
    ("splash_draft_context_prompt_end_rows_total", &["draft_context", "prompt_end_rows"]),
    ("splash_draft_context_materialization_rows_total", &["draft_context", "materialization_rows"]),
    ("splash_draft_context_avoided_rows_total", &["draft_context", "avoided_rows"]),
    ("splash_draft_state_restore_skipped_total", &["draft_context", "restore_skipped"]),
    ("splash_draft_state_resets_total", &["draft_context", "resets"]),
    ("splash_constraint_mask_overlap_batches_total", &["constraint_masks", "overlap_batches"]),
    ("splash_constraint_mask_overlap_requests_total", &["constraint_masks", "overlap_requests"]),
    ("splash_constraint_mask_target_forward_gpu_milliseconds", &["constraint_masks", "last_target_forward_gpu_ms"]),
    ("splash_constraint_mask_residual_wait_milliseconds", &["constraint_masks", "last_residual_wait_ms"]),
    ("splash_image_encodes_total", &["images", "encodes"]),
    ("splash_image_embedding_reuses_total", &["images", "embedding_reuses"]),
    ("splash_image_embedding_evictions_total", &["images", "embedding_evictions"]),
    ("splash_image_arena_bytes", &["images", "arena_bytes"]),
    ("splash_image_cached_bytes", &["images", "cached_bytes"]),
    ("splash_memory_current_bytes", &["memory_actual", "current_bytes"]),
    ("splash_memory_peak_bytes", &["memory_actual", "peak_bytes"]),
    ("splash_memory_denied_reservations_total", &["memory_governor", "denied_reservations"]),
    ("splash_memory_limit_bytes", &["memory_governor", "limit_bytes"]),
    ("splash_memory_headroom_bytes", &["memory_governor", "headroom_bytes"]),
    ("splash_admission_waiting_memory", &["admission", "waiting_memory"]),
    ("splash_admission_waiting_concurrency", &["admission", "waiting_concurrency"]),
    ("splash_admission_held_behind_refusal", &["admission", "held_behind_refusal"]),
    ("splash_admission_restoring", &["admission", "restoring"]),
    ("splash_admission_suspended", &["admission", "suspended"]),
    ("splash_admission_oldest_wait_milliseconds", &["admission", "oldest_wait_ms"]),
    ("splash_ttft_p50_milliseconds", &["metrics", "ttft_ms", "p50"]),
    ("splash_ttft_p95_milliseconds", &["metrics", "ttft_ms", "p95"]),
    ("splash_itl_p50_milliseconds", &["metrics", "itl_ms", "p50"]),
    ("splash_itl_p95_milliseconds", &["metrics", "itl_ms", "p95"]),
    ("splash_prefill_input_tokens_total", &["metrics", "prefill_input_tokens"]),
    ("splash_prefill_wall_milliseconds_total", &["metrics", "prefill_wall_ms"]),
    ("splash_prefill_tokens_per_second", &["metrics", "prefill_tokens_per_second"]),
    ("splash_decode_output_tokens_total", &["metrics", "decode_output_tokens"]),
    ("splash_decode_wall_milliseconds_total", &["metrics", "decode_wall_ms"]),
    ("splash_decode_cycle_milliseconds_total", &["metrics", "decode_cycle_ms"]),
    ("splash_decode_tokens_per_second", &["metrics", "decode_tokens_per_second"]),
    ("splash_drafted_tokens_total", &["metrics", "drafted_tokens"]),
    ("splash_accepted_draft_tokens_total", &["metrics", "accepted_draft_tokens"]),
    ("splash_draft_acceptance_ratio", &["metrics", "draft_acceptance_rate"]),
    ("splash_capacity_failures_total", &["metrics", "capacity_failures"]),
    ("splash_metal_failures_total", &["metrics", "metal_failures"]),
    ("splash_response_store_entries", &["response_store", "entries"]),
    ("splash_response_store_bytes", &["response_store", "bytes"]),
    ("splash_prefill_duty_percent", &["throttle", "duty_percent"]),
    ("splash_prefill_thermal_scale", &["throttle", "thermal_scale"]),
    ("splash_prefill_paced_waits_total", &["throttle", "paced_waits"]),
    ("splash_prefill_paced_wait_milliseconds", &["throttle", "paced_wait_ms"]),
];

const LATENCY_STAGES: &[&str] = &[
    "http_request", "upload", "preparation_queue", "preparation", "template",
    "tokenization", "grammar", "images", "native_queue", "http_ttft", "output_interval",
];
const LATENCY_BUCKETS: &[&str] = &[
    "0.001", "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1",
    "2.5", "5", "10", "30", "60", "120", "300", "900", "1800", "+Inf",
];
const LATENCY_BUCKET_SECONDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    30.0, 60.0, 120.0, 300.0, 900.0, 1800.0,
];

/// Fixed-size latency histograms compatible with Python's `latency.py`.
#[derive(Default)]
pub struct LatencyMetrics {
    histograms: Mutex<Vec<LatencyHistogram>>,
}

#[derive(Default)]
struct LatencyHistogram {
    counts: [u64; 19],
    sum: f64,
}

impl LatencyMetrics {
    pub fn timer(&self, stage: &'static str) -> LatencyTimer<'_> {
        LatencyTimer {
            metrics: self,
            stage,
            started: std::time::Instant::now(),
        }
    }

    pub fn observe(&self, stage: &str, elapsed: Duration) {
        let Some(index) = LATENCY_STAGES.iter().position(|candidate| *candidate == stage) else {
            return;
        };
        let seconds = elapsed.as_secs_f64();
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        let mut histograms = self.histograms.lock().expect("latency metrics lock");
        if histograms.is_empty() {
            histograms.resize_with(LATENCY_STAGES.len(), LatencyHistogram::default);
        }
        let histogram = &mut histograms[index];
        let bucket = LATENCY_BUCKET_SECONDS
            .iter()
            .position(|bound| seconds <= *bound)
            .unwrap_or(18);
        histogram.counts[bucket] += 1;
        histogram.sum += seconds;
    }

    pub fn snapshot(&self) -> Value {
        let histograms = self.histograms.lock().expect("latency metrics lock");
        let mut snapshot = serde_json::Map::new();
        for (index, stage) in LATENCY_STAGES.iter().enumerate() {
            let histogram = histograms.get(index);
            let mut cumulative = 0u64;
            let mut buckets = serde_json::Map::new();
            for (bucket_index, bound) in LATENCY_BUCKETS.iter().enumerate() {
                cumulative += histogram.map(|histogram| histogram.counts[bucket_index]).unwrap_or(0);
                buckets.insert((*bound).to_owned(), Value::from(cumulative));
            }
            snapshot.insert(
                (*stage).to_owned(),
                serde_json::json!({
                    "buckets": buckets,
                    "count": cumulative,
                    "sum": histogram.map(|histogram| histogram.sum).unwrap_or(0.0),
                }),
            );
        }
        Value::Object(snapshot)
    }
}

pub struct LatencyTimer<'a> {
    metrics: &'a LatencyMetrics,
    stage: &'static str,
    started: std::time::Instant,
}

impl Drop for LatencyTimer<'_> {
    fn drop(&mut self) {
        self.metrics.observe(self.stage, self.started.elapsed());
    }
}

/// Render the same low-cardinality gauges/counters and optional latency
/// histograms that the Python endpoint exposes.
pub fn prometheus_metrics(status: &Value) -> String {
    let mut lines = vec![
        "# HELP splash_info Splash runtime metrics.".to_owned(),
        "# TYPE splash_info gauge".to_owned(),
        "splash_info{runtime=\"native\"} 1".to_owned(),
    ];
    for (field, states) in [
        ("memory_pressure", &["normal", "warning", "critical"][..]),
        ("thermal_state", &["nominal", "fair", "serious", "critical"][..]),
    ] {
        let current = status.get(field).and_then(Value::as_str).unwrap_or("");
        for state in states {
            lines.push(format!(
                "splash_{}{{state=\"{}\"}} {}",
                if field == "memory_pressure" { "memory_pressure" } else { "thermal_state" },
                state,
                u8::from(current == *state)
            ));
        }
    }
    if let Some(speeds) = status.pointer("/fans/speeds_rpm").and_then(Value::as_array) {
        for (index, speed) in speeds.iter().enumerate() {
            if let Some(number) = finite_number(speed) {
                lines.push(format!("splash_fan_rpm{{fan=\"{index}\"}} {number}"));
            }
        }
    }
    if let Some(temperatures) = status.pointer("/fans/temperatures").and_then(Value::as_array) {
        for temperature in temperatures {
            let sensor = temperature.get("sensor").and_then(Value::as_str).unwrap_or("");
            if !sensor.is_empty() {
                if let Some(number) = temperature.get("celsius").and_then(finite_number) {
                    lines.push(format!(
                        "splash_temperature_celsius{{sensor=\"{}\"}} {number}",
                        escape_label(sensor)
                    ));
                }
            }
        }
    }
    let gated = status.pointer("/disk/writes_gated").and_then(Value::as_str).unwrap_or("");
    for state in ["no", "budget", "closed"] {
        lines.push(format!("splash_disk_writes_gated{{state=\"{state}\"}} {}", u8::from(gated == state)));
    }
    for (name, path) in SERIES {
        if let Some(value) = lookup(status, path).and_then(metric_number) {
            lines.push(format!("{name} {value}"));
        }
    }
    for stage in LATENCY_STAGES {
        let Some(histogram) = status.get("latency").and_then(|v| v.get(*stage)) else { continue };
        let name = format!("splash_{stage}_seconds");
        lines.push(format!("# TYPE {name} histogram"));
        if let Some(buckets) = histogram.get("buckets") {
            for bound in LATENCY_BUCKETS {
                if let Some(value) = buckets.get(*bound).and_then(finite_number) {
                    lines.push(format!("{name}_bucket{{le=\"{bound}\"}} {value}"));
                }
            }
        }
        if let Some(value) = histogram.get("count").and_then(finite_number) {
            lines.push(format!("{name}_count {value}"));
        }
        if let Some(value) = histogram.get("sum").and_then(finite_number) {
            lines.push(format!("{name}_sum {value}"));
        }
    }
    lines.join("\n") + "\n"
}

fn lookup<'a>(status: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(status, |value, key| value.get(*key))
}

fn finite_number(value: &Value) -> Option<f64> {
    let number = value.as_f64()?;
    number.is_finite().then_some(number)
}

fn metric_number(value: &Value) -> Option<f64> {
    if let Some(boolean) = value.as_bool() {
        return Some(f64::from(u8::from(boolean)));
    }
    finite_number(value)
}

fn escape_label(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\n', "\\n").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_python_series_enum_gauges_and_latency_histograms() {
        let status = json!({
            "ready": true,
            "memory_pressure": "warning",
            "thermal_state": "serious",
            "fans": {"speeds_rpm": [1200, "bad", false], "temperatures": [
                {"sensor": "CPU\" Pro", "celsius": 64.5}, {"sensor": "bad", "celsius": null}
            ]},
            "disk": {"writes_gated": "budget"},
            "requests": {"submitted": 7},
            "metrics": {"decode_tokens_per_second": 12.5},
            "latency": {"http_request": {"buckets": {"0.001": 2, "+Inf": 3}, "count": 3, "sum": 0.4}}
        });
        let rendered = prometheus_metrics(&status);
        assert!(rendered.contains("splash_ready 1\n"));
        assert!(rendered.contains("splash_memory_pressure{state=\"warning\"} 1"));
        assert!(rendered.contains("splash_thermal_state{state=\"serious\"} 1"));
        assert!(rendered.contains("splash_fan_rpm{fan=\"0\"} 1200"));
        assert!(rendered.contains("sensor=\"CPU\\\" Pro\""));
        assert!(rendered.contains("splash_disk_writes_gated{state=\"budget\"} 1"));
        assert!(rendered.contains("splash_requests_submitted_total 7"));
        assert!(rendered.contains("splash_decode_tokens_per_second 12.5"));
        assert!(rendered.contains("splash_http_request_seconds_bucket{le=\"+Inf\"} 3"));
        assert!(rendered.ends_with('\n'));
    }

    #[test]
    fn converts_boolean_series_and_ignores_missing_or_non_finite_values() {
        let rendered = prometheus_metrics(&json!({"requests": {"failed": true}, "metrics": {"decode_tokens_per_second": null}}));
        assert!(rendered.contains("splash_requests_failed_total 1"));
        assert!(!rendered.contains("splash_decode_tokens_per_second"));
    }

    #[test]
    fn latency_snapshot_has_python_buckets_and_observes_cumulatively() {
        let metrics = LatencyMetrics::default();
        metrics.observe("http_request", Duration::from_millis(7));
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["http_request"]["count"], 1);
        assert_eq!(snapshot["http_request"]["buckets"]["0.005"], 0);
        assert_eq!(snapshot["http_request"]["buckets"]["0.01"], 1);
        assert_eq!(snapshot["upload"]["count"], 0);
    }

    #[test]
    fn timers_record_their_stage_once_when_dropped() {
        let metrics = LatencyMetrics::default();
        {
            let _timer = metrics.timer("upload");
        }
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["upload"]["count"], 1);
        assert!(snapshot["upload"]["sum"].as_f64().unwrap() >= 0.0);
    }
}
