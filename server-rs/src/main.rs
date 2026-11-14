use std::error::Error;
use std::fs::{self, File};
use std::net::{IpAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use splash_server::chat_format::{flint, qwen};
use splash_server::engine::EngineClient;
use splash_server::serve::{Pipeline, Server};
use splash_server::templates::load_chat_templates;
use splash_server::tokenization::{FastTokenizer, PromptTokenizer};

use splash_server::assembly;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn main() {
    let result = run();
    if STOP.load(Ordering::Relaxed) {
        splash_server::diagnostics::print_status("Stopping · releasing engine resources");
    }
    splash_server::diagnostics::close_console();
    if let Err(error) = result {
        eprintln!("splash-server: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("--help" | "-h") => {
            print_help();
            return Ok(());
        }
        Some("--version") => {
            println!("splash-server {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some("model-path") => {
            let model = args.next().ok_or("model-path requires a model ID")?;
            if args.next().is_some() {
                return Err("model-path takes one model ID".into());
            }
            println!("{}", find_model(&model)?.display());
            return Ok(());
        }
        Some("inspect-gguf") => {
            let path = args.next().ok_or("inspect-gguf requires a GGUF file")?;
            let include_tensors = match args.next().as_deref() {
                None => false,
                Some("--tensors") => true,
                Some(_) => return Err("inspect-gguf accepts only optional --tensors".into()),
            };
            if args.next().is_some() { return Err("inspect-gguf accepts only optional --tensors".into()); }
            let metadata = splash_server::gguf::Metadata::read(&path, include_tensors)?;
            let tensors = metadata.tensors.iter().map(|(name, tensor)| (name.clone(), serde_json::json!({
                "extents": tensor.extents, "kind": tensor.kind, "offset": tensor.offset
            }))).collect::<serde_json::Map<_, _>>();
            let mut output = serde_json::Map::new();
            output.insert("metadata".into(), serde_json::to_value(&metadata.values)?);
            output.insert("tensor_count_read".into(), Value::from(tensors.len()));
            if include_tensors { output.insert("tensors".into(), Value::Object(tensors)); }
            println!("{}", serde_json::to_string_pretty(&Value::Object(output))?);
            return Ok(());
        }
        Some("gguf-config") => {
            run_gguf_config(args.collect())?;
            return Ok(());
        }
        Some("prepare-gguf") => {
            run_prepare_gguf(args.collect())?;
            return Ok(());
        }
        Some("prepare-mlx") => {
            run_prepare_mlx(args.collect())?;
            return Ok(());
        }
        Some("install-cached") => {
            run_install_cached(args.collect())?;
            return Ok(());
        }
        Some("document-worker") => {
            if args.next().as_deref() != Some("--pdfium") {
                return Err("document-worker requires --pdfium PATH".into());
            }
            let path = args.next().ok_or("document-worker requires a PDFium library path")?;
            if args.next().is_some() {
                return Err("document-worker takes only --pdfium PATH".into());
            }
            splash_server::documents::run_pdfium_worker(PathBuf::from(path))?;
            return Ok(());
        }
        Some("serve") => {}
        Some(other) => return Err(format!("unknown command: {other}").into()),
    }

    let mut model = None;
    let mut directory = None;
    let mut draft = None;
    let mut mmproj = None;
    let mut language_only = false;
    let mut speculation = Speculation::Auto;
    let mut require_draft = false;
    let mut port = 8000;
    let mut host: IpAddr = "127.0.0.1".parse()?;
    let mut allowed_hosts = Vec::new();
    let mut allowed_origins = Vec::new();
    let mut request_timeout = None;
    let mut max_request_size = splash_server::http::DEFAULT_MAX_REQUEST_SIZE;
    let mut thinking_token_budget = std::env::var("SPLASH_THINKING_TOKEN_BUDGET")
        .ok()
        .map(|value| parse_thinking_budget(&value))
        .transpose()?
        .flatten();
    let mut compact_at = None;
    let mut compact_keep = None;
    let mut compact_memory = None;
    let mut max_image_pixels = splash_server::images::MAX_PIXELS;
    let mut pdfium_library = std::env::var_os("SPLASH_PDFIUM_PATH").map(PathBuf::from);
    let mut binary = None;
    let mut pending = 8;
    let mut no_webui = false;
    let mut serve_record_path: Option<PathBuf> = None;
    let mut served_model_names = Vec::new();
    let mut announce_served_name = false;
    let mut default_reasoning_effort = std::env::var("SPLASH_DEFAULT_REASONING_EFFORT")
        .ok().map(|value| parse_reasoning_effort(&value)).transpose()?;
    let mut loop_guard = false;
    let mut thinking_end_text = std::env::var("SPLASH_THINKING_END_TEXT").ok();
    let mut expert_memory_grow = None;
    let mut disable_ane = false;
    let mut disable_ane_mixer = false;
    let ane_share = std::env::var("SPLASH_ANE_SHARE").ok();
    let mut context = "auto".to_owned();
    let mut memory = "auto".to_owned();
    let mut disk = None;
    let mut max_cache_disk_writes = 0u64;
    let mut persistent_cache = false;
    let mut cache_dir = None;
    let mut thermal_policy = "auto".to_owned();
    let mut prefill_duty = None;
    let mut engine_args = Vec::new();
    let mut key = std::env::var("SPLASH_API_KEY").ok().filter(|key| !key.is_empty());
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--require-draft" => require_draft = true,
            "--language-only" => language_only = true,
            "--disable-ane" => disable_ane = true,
            "--disable-ane-mixer" => disable_ane_mixer = true,
            "--no-webui" => no_webui = true,
            "--serve-record" => serve_record_path = Some(PathBuf::from(args.next().ok_or("--serve-record requires a path")?)),
            "--announce-served-name" => announce_served_name = true,
            "--loop-guard" => loop_guard = true,
            "--persistent-cache" => persistent_cache = true,
            "--allow-idle-sleep" => engine_args.extend(["--idle-sleep".into(), "allow".into()]),
            _ => {
                let value = args.next().ok_or_else(|| format!("{flag} requires a value"))?;
                match flag.as_str() {
                    "--model" => model = Some(value),
                    "--model-dir" => directory = Some(PathBuf::from(value)),
                    "--draft-model" => draft = Some(value),
                    "--mmproj" => mmproj = Some(value),
                    "--speculation" => speculation = Speculation::parse(&value)?,
                    "--binary" => binary = Some(PathBuf::from(value)),
                    "--pdfium" => pdfium_library = Some(PathBuf::from(value)),
                    "--port" => port = value.parse()?,
                    "--request-timeout" => {
                        request_timeout = Some(Duration::try_from_secs_f64(
                            parse_positive_duration(&value, "--request-timeout")?,
                        )?);
                    }
                    "--max-request-size" => {
                        max_request_size = parse_request_size(&value)?;
                    }
                    "--thinking-token-budget" => {
                        thinking_token_budget = parse_thinking_budget(&value)?;
                    }
                    "--compact-at" => compact_at = Some(parse_compact_at(&value)?),
                    "--compact-keep" => {
                        let keep = value.parse::<u32>()?;
                        if keep == 0 {
                            return Err("--compact-keep must be positive".into());
                        }
                        compact_keep = Some(keep);
                    }
                    "--compact-memory" => {
                        if value.is_empty() {
                            return Err("--compact-memory requires a directory path".into());
                        }
                        compact_memory = Some(PathBuf::from(value));
                    }
                    "--served-model-name" => {
                        validate_served_model_name(&value)?;
                        served_model_names.push(value);
                    }
                    "--default-reasoning-effort" => {
                        default_reasoning_effort = Some(parse_reasoning_effort(&value)?);
                    }
                    "--thinking-end-text" => thinking_end_text = Some(value),
                    "--expert-memory-grow" => {
                        expert_memory_grow = Some(parse_expert_memory_grow(&value)?);
                    }
                    "--max-image-pixels" => {
                        max_image_pixels = value.parse().map_err(|_| {
                            format!("--max-image-pixels must be an integer between {} and {}",
                                splash_server::images::MIN_PIXELS, splash_server::images::MAX_PIXELS)
                        })?;
                        if !(splash_server::images::MIN_PIXELS..=splash_server::images::MAX_PIXELS)
                            .contains(&max_image_pixels)
                        {
                            return Err(format!(
                                "--max-image-pixels must be between {} and {}",
                                splash_server::images::MIN_PIXELS, splash_server::images::MAX_PIXELS
                            ).into());
                        }
                    }
                    "--queue-size" => pending = value.parse()?,
                    "--api-key" => key = Some(value),
                    "--host" => host = value.parse()?,
                    "--allowed-host" => {
                        let (host, port) = splash_server::http::security::parse_authority(&value)
                            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
                        if port.is_some() {
                            return Err("--allowed-host takes a host name without a port".into());
                        }
                        allowed_hosts.push(host);
                    }
                    "--allowed-origin" => {
                        allowed_origins.push(
                            splash_server::http::parse_allowed_origin(&value)
                                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?,
                        );
                    }
                    "--max-context" => context = parse_context(&value)?,
                    "--max-memory" => memory = parse_size_bytes(&value, true)?
                        .map(|size| size.to_string()).unwrap_or_else(|| "auto".into()),
                    "--max-cache-disk" => disk = parse_cache_disk(&value)?,
                    "--max-cache-disk-writes" => max_cache_disk_writes = parse_cache_writes(&value)?,
                    "--ple-memory" | "--expert-memory" => {
                        if let Some(value) = parse_memory_option(&value, true)? {
                            engine_args.extend([flag, value]);
                        }
                    }
                    "--sparse-kv-memory" => {
                        if let Some(value) = parse_memory_option(&value, false)? {
                            engine_args.extend([flag, value]);
                        }
                    }
                    "--idle-release" | "--cache-idle-spill" | "--residency-keepalive" => {
                        let parsed = parse_idle_duration(&value, &flag)?;
                        engine_args.extend([flag, parsed]);
                    }
                    "--cache-dir" => {
                        let path = resolve_cache_dir(&value)?;
                        cache_dir = Some(path.clone());
                        engine_args.extend([flag, path.to_string_lossy().into_owned()]);
                    }
                    "--cache-eviction-protect" => {
                        let count = parse_cache_eviction_protect(&value)?;
                        if count > 0 {
                            engine_args.extend([flag, count.to_string()]);
                        }
                    }
                    "--decode-share" => {
                        let parsed = parse_bounded_float(&value, &flag, 0.0, None)?;
                        engine_args.extend([flag, parsed]);
                    }
                    "--prefill-slice-ms" => {
                        let parsed = parse_bounded_float(&value, &flag, 1.0, Some(10_000.0))?;
                        engine_args.extend([flag, parsed]);
                    }
                    "--prefill-duty" => {
                        prefill_duty = Some(parse_bounded_float(&value, &flag, 10.0, Some(100.0))?);
                    }
                    "--kv-format" => {
                        validate_choice(&value, &["int8", "int4", "k8v4", "bf16"], &flag)?;
                        engine_args.extend([flag, value]);
                    }
                    "--state-format" => {
                        validate_choice(&value, &["fp32", "bf16"], &flag)?;
                        engine_args.extend([flag, value]);
                    }
                    "--cache-idle-evict" => {
                        validate_choice(&value, &["on", "off"], &flag)?;
                        engine_args.extend([flag, value]);
                    }
                    "--cache-eviction" => {
                        validate_choice(&value, &["lru", "fifo", "lfu", "largest"], &flag)?;
                        if value != "lru" {
                            engine_args.extend([flag, value]);
                        }
                    }
                    "--verify-rows" => {
                        validate_choice(&value, &["auto", "all"], &flag)?;
                        if value != "auto" {
                            engine_args.extend([flag, value]);
                        }
                    }
                    "--thermal-policy" => {
                        validate_choice(&value, &["auto", "balanced", "moderate", "quiet"], &flag)?;
                        thermal_policy = value;
                    }
                    _ => return Err(format!("option {flag} is not implemented in the Rust server").into()),
                }
            }
        }
    }
    if pending == 0 {
        return Err("--queue-size must be positive".into());
    }
    validate_announce_served_name(announce_served_name, &served_model_names)?;
    validate_cache_dependencies(
        persistent_cache,
        cache_dir.is_some(),
        disk.as_deref(),
        max_cache_disk_writes,
    )?;
    if let Some(share) = ane_share.as_deref() {
        if disable_ane {
            return Err("SPLASH_ANE_SHARE cannot be combined with --disable-ane".into());
        }
        engine_args.extend(["--ane-share".into(), parse_ane_share(share)?]);
    }
    if disable_ane {
        engine_args.extend(["--ane".into(), "off".into()]);
    }
    if max_cache_disk_writes > 0 {
        engine_args.extend([
            "--max-cache-disk-writes".into(),
            max_cache_disk_writes.to_string(),
        ]);
    }
    if persistent_cache {
        engine_args.extend(["--persistent-cache".into(), "on".into()]);
    }
    if let Some(duty) = resolved_prefill_duty(&thermal_policy, prefill_duty) {
        engine_args.extend(["--prefill-duty".into(), duty]);
    }
    engine_args.extend(engine_parity_args(
        expert_memory_grow.as_deref(),
        disable_ane_mixer,
        loop_guard,
    ));
    if compact_keep.is_some() && compact_at.is_none() {
        return Err("--compact-keep needs --compact-at".into());
    }
    if compact_memory.is_some() && compact_at.is_none() {
        return Err("--compact-memory needs --compact-at".into());
    }
    if let Some(key) = &key {
        splash_server::http::validate_api_key(key)?;
    }
    if !host.is_loopback() && key.is_none() {
        return Err("binding beyond loopback requires --api-key or SPLASH_API_KEY".into());
    }
    splash_server::diagnostics::start_console()?;
    let serve_record = splash_server::serve_record::ServeRecord::new(serve_record_path);
    serve_record.starting()?;
    let matches_draft = |record: &Value| {
        let source = record.pointer("/sources/draft/repo").and_then(Value::as_str);
        let provider = record.pointer("/speculation/provider").and_then(Value::as_str);
        let has_draft = match provider {
            Some(provider) => provider == "dflash2",
            None => source.is_some() && record["version"] == 1,
        };
        match (&draft, speculation) {
            (Some(draft), _) => has_draft && source.is_some_and(|source| same_source(source, draft)),
            (None, Speculation::Dflash2) => has_draft,
            (None, Speculation::Auto) => !require_draft || has_draft,
        }
    };
    let selection = draft.as_ref().map(|draft| format!(" with draft {draft}")).unwrap_or_default();
    let models = models_root()?;
    let path = match (directory, model.as_deref()) {
        (Some(path), _) if path.join("model.json").is_file() => path,
        (Some(path), _) => {
            // The installer records a local target's assembly under the
            // selection link its options hash to, so the same directory may
            // hold several prepared assemblies: the link names the one this
            // command asks for, and only a selection the installer never
            // recorded falls back to matching the records themselves.
            let path = path.canonicalize()?;
            let target = path.to_string_lossy().into_owned();
            // A local DFlash2 path fully identifies the prepared draft. The
            // `auto` and explicit `dflash2` modes therefore refer to the same
            // local assembly; hashing the mode separately makes a valid GGUF
            // selection disappear whenever the UI toggles `auto` to
            // `dflash2`, especially for directories with multiple targets.
            let link_speculation = if draft.is_some() && speculation == Speculation::Dflash2 {
                Speculation::Auto
            } else {
                speculation
            };
            let link = local_selection_link(
                &models,
                &target,
                language_only,
                draft.as_deref(),
                mmproj.as_deref(),
                link_speculation,
                require_draft,
            );
            match assembly_at(&link) {
                Some(assembly) => assembly,
                None => find_assembly(
                    |record| matches_draft(record) && record.pointer("/sources/target/repo")
                        .and_then(Value::as_str).is_some_and(|source| same_source(source, &target)),
                    &format!("{target}{selection}"),
                )?,
            }
        }
        (None, Some(model)) => {
            let (repo, variant) = split_model_id(model);
            let link = match variant {
                Some(variant) => models.join(format!("{repo}:{variant}")),
                None => models.join(&repo),
            };
            match assembly_at(&link) {
                Some(assembly) => assembly,
                None => find_assembly(
                    |record| matches_model(record, model) && matches_draft(record),
                    &format!("{model}{selection}"),
                )?,
            }
        }
        _ => return Err("specify --model or --model-dir".into()),
    };
    let path = path.canonicalize()?;
    let record = File::open(path.join("model.json"))
        .map_err(|error| format!("{} is not a prepared assembly: {error}", path.display()))?;
    record.lock_shared()?;
    let assembly: Value = serde_json::from_reader(&record)?;
    let full_assembly_verification = match std::env::var("SPLASH_VERIFY_ASSEMBLY") {
        Ok(value) if value == "full" => true,
        Ok(value) if value == "fast" || value.is_empty() => false,
        Ok(_) => return Err("SPLASH_VERIFY_ASSEMBLY must be fast or full".into()),
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    assembly::verify(&path, &assembly, full_assembly_verification)?;
    if !matches_draft(&assembly) {
        return Err(format!("{} does not match the requested speculation settings{selection}", path.display()).into());
    }
    let format = match assembly["family"].as_str() {
        Some("Flint") => flint(),
        Some("Qwen3.8-4B" | "Qwen3.5-9B" | "Qwen3.8-27B" | "Qwen3.6-35B-A3B"
            | "Qwen3.5-122B-A10B" | "Qwen3.8-Flash-Next") => qwen(),
        family => return Err(format!("unsupported model family: {family:?}").into()),
    };
    let model = model.or_else(|| assembly["model"].as_str().map(str::to_owned))
        .ok_or("assembly does not name its model")?;
    let listener = match TcpListener::bind((host, port)) {
        Ok(listener) => listener,
        Err(error) => {
            serve_record.failed(splash_server::serve_record::FailureKind::Port, "unable to bind HTTP listener");
            return Err(error.into());
        }
    };
    serve_record.entered("loading");
    splash_server::diagnostics::print_status(&format!("Loading · {model}"));
    let tokenizer_directory = path.join("tokenizer");
    let tokenizer = Arc::new(FastTokenizer::from_file(&tokenizer_directory)?);
    let templates = load_chat_templates(&tokenizer_directory)?;
    let make_prompt_tokenizer = || {
        PromptTokenizer::from_directory(
            tokenizer.clone(),
            &tokenizer_directory,
            format.message_boundary,
        )
    };
    // Rebuild the constraint factory for each engine generation: an old
    // pipeline is never reused after its client pipes fail.
    let make_constraints = || {
        splash_server::constraints::ConstraintTokenizer::from_directory(
            &tokenizer_directory,
            format.vocabulary,
            &format.stop_ids(),
        )
        .ok()
        .and_then(|env| splash_server::constraints::ConstraintFactory::new(&env, &format).ok())
    };
    let binary = match binary {
        Some(binary) => binary,
        None => std::env::current_exe()?.with_file_name("splash"),
    };
    let documents = if let Some(pdfium_library) = pdfium_library {
        if !pdfium_library.is_file() {
            return Err(format!("PDFium library does not exist: {}", pdfium_library.display()).into());
        }
        Some(Arc::new(splash_server::documents::DocumentService::new(Arc::new(
            splash_server::documents::ProcessPdfRenderer::new(
                std::env::current_exe()?,
                pdfium_library,
            ),
        ))))
    } else {
        None
    };
    let thinking_key = splash_server::thinking::load_thinking_key(None)?;
    let thinking_codec = Arc::new(splash_server::thinking::ThinkingCodec::new(&thinking_key)?);
    let engine_command = || {
        let mut command = Command::new(&binary);
        command
            .arg("serve-native")
            .arg(&path)
            .args([context.as_str(), memory.as_str()]);
        if let Some(disk) = &disk {
            command.arg(disk);
        }
        command.args(&engine_args);
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        command
    };

    unsafe {
        libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
    }
    const MAX_ENGINE_RESTARTS: u32 = 3;
    let mut restart_count = 0;
    let restart_total = Arc::new(AtomicU64::new(0));
    loop {
        if STOP.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut child = match engine_command().spawn() {
            Ok(child) => EngineProcess(child),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                serve_record.failed(splash_server::serve_record::FailureKind::Engine, "native engine executable is unavailable");
                return Err(format!("native engine executable is missing: {error}").into());
            }
            Err(error) => {
                let delay = recorded_restart_delay(
                    &mut restart_count, &error.to_string(), MAX_ENGINE_RESTARTS, &serve_record,
                    "native engine failed after restart limit",
                )?;
                restart_total.fetch_add(1, Ordering::Relaxed);
                if !wait_restart_backoff(delay) {
                    return Ok(());
                }
                continue;
            }
        };
        let engine_client = EngineClient::over_streams(
            Box::new(child.0.stdout.take().expect("piped stdout")),
            Box::new(child.0.stdin.take().expect("piped stdin")),
            pending,
        );
        let mut startup_failure = None;
        let startup_started = Instant::now();
        while !engine_client.wait_ready(Some(Duration::from_millis(100))) {
            if STOP.load(Ordering::Relaxed) {
                engine_client.close();
                drop(child);
                return Ok(());
            }
            if let Some(error) = engine_client.fatal_error() {
                startup_failure = Some(error.to_string());
                break;
            }
            match child.0.try_wait() {
                Ok(Some(status)) => {
                    startup_failure = Some(format!("engine exited during startup: {status}"));
                    break;
                }
                Err(error) => {
                    startup_failure = Some(format!("could not inspect engine process: {error}"));
                    break;
                }
                Ok(None) => {}
            }
            if startup_started.elapsed() >= Duration::from_secs(180) {
                startup_failure = Some("engine did not become ready within 180 seconds".into());
                break;
            }
        }
        if let Some(reason) = startup_failure {
            engine_client.close();
            drop(child);
            if STOP.load(Ordering::Relaxed) {
                return Ok(());
            }
            let delay = recorded_restart_delay(
                &mut restart_count, &reason, MAX_ENGINE_RESTARTS, &serve_record,
                "native engine failed during startup",
            )?;
            restart_total.fetch_add(1, Ordering::Relaxed);
            if !wait_restart_backoff(delay) {
                return Ok(());
            }
            continue;
        }

        if restart_total.load(Ordering::Relaxed) > 0 {
            splash_server::diagnostics::print_status("Engine restarted");
        }
        serve_record.ready()?;
        splash_server::diagnostics::print_status(&format!("Ready · {model} · http://{}", listener.local_addr()?));
        let ready_since = Instant::now();
        let monitor_client = engine_client.clone();
        let server = Server::new(Pipeline {
            tokenizer: tokenizer.clone(),
            prompt_tokenizer: make_prompt_tokenizer(),
            templates: templates.clone(),
            chat_format: format.clone(),
            client: engine_client,
            model: model.clone(),
            images: splash_server::images::ImageCache::new(),
            max_image_pixels,
            default_reasoning_effort: default_reasoning_effort.clone(),
            thinking_end_tokens: thinking_end_tokens(
                &tokenizer,
                thinking_end_text.as_deref(),
                format.think_end_id,
            )?,
            documents: documents.clone(),
            constraints: make_constraints(),
        })
        .with_api_key(key.clone())
        .with_thinking_codec(thinking_codec.clone())
        .with_request_timeout(request_timeout)
        .with_max_request_size(max_request_size)
        .with_thinking_token_budget(thinking_token_budget)
        .with_loop_guard(loop_guard)
        .with_webui(!no_webui)
        .with_served_model_names(served_model_names.clone(), announce_served_name)
        .with_compaction(compact_at, compact_keep, compact_memory.clone())
        .with_restart_count(restart_total.clone())
        .with_connection_limit(pending.saturating_add(64))
        .with_allowed_hosts(
            std::iter::once(host.to_string())
                .chain(allowed_hosts.iter().cloned())
                .collect(),
        )
        .with_allowed_origins(allowed_origins.clone());

        let mut engine_exited = None;
        let mut process_error = None;
        let mut engine_failure = None;
        let serve_result = server.serve_until(listener.try_clone()?, || {
            if STOP.load(Ordering::Relaxed) {
                return true;
            }
            if let Some(error) = monitor_client.fatal_error() {
                engine_failure = Some(error);
                return true;
            }
            match child.0.try_wait() {
                Ok(Some(status)) => engine_exited = Some(status),
                Err(error) => process_error = Some(error),
                Ok(None) => {}
            }
            engine_exited.is_some() || process_error.is_some()
        });
        monitor_client.close();
        drop(child);
        serve_result?;
        if STOP.load(Ordering::Relaxed) {
            return Ok(());
        }
        let reason = engine_failure
            .map(|error| error.to_string())
            .or_else(|| engine_exited.map(|status| format!("engine exited: {status}")))
            .or_else(|| process_error.map(|error| format!("could not inspect engine process: {error}")))
            .unwrap_or_else(|| "HTTP listener stopped unexpectedly".into());
        if ready_since.elapsed() >= Duration::from_secs(600) {
            restart_count = 0;
        }
        let delay = recorded_restart_delay(
            &mut restart_count, &reason, MAX_ENGINE_RESTARTS, &serve_record,
            "native engine failed after restart limit",
        )?;
        restart_total.fetch_add(1, Ordering::Relaxed);
        if !wait_restart_backoff(delay) {
            return Ok(());
        }
    }
}

fn print_help() {
    println!("Usage: splash-server serve (--model ID | --model-dir PATH) [OPTIONS]");
    println!("       splash-server model-path ID");
    println!("       splash-server inspect-gguf FILE [--tensors]");
    println!("       splash-server gguf-config FILE [--vision FILE] [--engine PATH]");
    println!("       splash-server prepare-gguf TARGET --models-dir DIR [--vision FILE] [--engine PATH] [--model-id NAME]");
    println!("         Prepares one local GGUF (single or complete split set), optionally with one mmproj; draft models are unsupported.");
    println!("       splash-server prepare-mlx DIRECTORY --models-dir DIR [--engine PATH] [--model-id NAME] [--language-only]");
    println!("         Prepares a local MLX-affine model with one model.safetensors checkpoint.");
    println!("       splash-server install-cached [--cache-dir DIR] [--models-dir DIR] [--engine PATH]");
    println!("         Installs supported single or split GGUFs already in the local Hugging Face cache (offline).");
    println!("\nModel selection:");
    println!("      --model ID                         Prepared model ID");
    println!("      --model-dir PATH                   Prepared assembly directory");
    println!("      --draft-model ID                  Draft model ID");
    println!("      --mmproj PATH                     Vision projector file");
    println!("      --speculation MODE                auto, off, or a supported draft mode");
    println!("      --require-draft                   Refuse to start without draft weights");
    println!("      --language-only                   Skip vision model preparation");
    println!("      --binary PATH                     Native engine executable");
    println!("\nHTTP and access:");
    println!("      --host IP                         Listen address (default 127.0.0.1)");
    println!("      --port PORT                       Listen port (default 8000)");
    println!("      --api-key KEY                     Protect API routes");
    println!("      --allowed-host NAME               Additional accepted Host name (repeatable)");
    println!("      --allowed-origin ORIGIN           Browser origin (repeatable)");
    println!("      --queue-size N                    Maximum pending engine requests");
    println!("      --request-timeout SECONDS         Global request deadline");
    println!("      --max-request-size SIZE           Request body limit (supports K/M/G)");
    println!("      --serve-record PATH               Atomic server lifecycle record");
    println!("      --no-webui                        Disable the bundled web UI");
    println!("\nGeneration and context:");
    println!("      --default-reasoning-effort VALUE  Default reasoning effort");
    println!("      --thinking-token-budget N         Thinking budget");
    println!("      --thinking-end-text TEXT           Thinking end marker");
    println!("      --loop-guard                      Enable engine loop guard");
    println!("      --max-context N|auto              Context size");
    println!("      --compact-at auto|FRACTION        Compact long conversations");
    println!("      --compact-keep TOKENS             Recent tokens to keep after compaction");
    println!("      --compact-memory DIR              Durable compaction note directory");
    println!("      --max-image-pixels N              Per-image resize budget");
    println!("      --pdfium PATH                     PDFium library for PDF inputs");
    println!("\nMemory, cache, and scheduling:");
    println!("      --max-memory SIZE                 Memory budget (supports auto/off/size)");
    println!("      --max-cache-disk SIZE             SSD cache budget");
    println!("      --max-cache-disk-writes SIZE      SSD cache write budget");
    println!("      --persistent-cache                Enable persistent SSD cache");
    println!("      --cache-dir DIR                   Persistent cache directory");
    println!("      --cache-idle-evict on|off         Idle cache eviction");
    println!("      --cache-eviction POLICY            lru, fifo, lfu, or largest");
    println!("      --cache-eviction-protect N         Protect recent cache entries");
    println!("      --idle-release DURATION            Release idle resources after duration");
    println!("      --cache-idle-spill DURATION        Spill idle cache after duration");
    println!("      --residency-keepalive DURATION     Keep model residency warm");
    println!("      --decode-share FRACTION            Decode scheduling share");
    println!("      --prefill-slice-ms MS              Prefill slice duration");
    println!("      --prefill-duty PERCENT             Prefill duty limit");
    println!("\nEngine options:");
    println!("      --ple-memory SIZE                  PLE memory budget");
    println!("      --expert-memory SIZE               Expert memory budget");
    println!("      --sparse-kv-memory SIZE            Sparse KV memory budget");
    println!("      --expert-memory-grow on|off         Expert memory growth");
    println!("      --kv-format FORMAT                 int8, int4, k8v4, or bf16");
    println!("      --state-format FORMAT              fp32 or bf16");
    println!("      --verify-rows auto|all             Row verification policy");
    println!("      --thermal-policy VALUE             auto, balanced, moderate, or quiet");
    println!("      --allow-idle-sleep                 Allow system idle sleep");
    println!("      --disable-ane                     Disable ANE execution");
    println!("      --disable-ane-mixer               Disable ANE mixer");
    println!("\nModel aliases:");
    println!("      --served-model-name NAME           Public model alias (repeatable)");
    println!("      --announce-served-name             Announce the public model alias");
}

fn recorded_restart_delay(
    restart_count: &mut u32,
    reason: &str,
    maximum: u32,
    record: &splash_server::serve_record::ServeRecord,
    safe_failure: &'static str,
) -> Result<Duration, Box<dyn Error>> {
    match next_restart_delay(restart_count, reason, maximum) {
        Ok(delay) => Ok(delay),
        Err(error) => {
            record.failed(splash_server::serve_record::FailureKind::Engine, safe_failure);
            Err(error)
        }
    }
}

fn next_restart_delay(
    restart_count: &mut u32,
    _reason: &str,
    maximum: u32,
) -> Result<Duration, Box<dyn Error>> {
    if *restart_count >= maximum {
        return Err(format!("native engine failed after {maximum} restart attempts").into());
    }
    *restart_count += 1;
    let delay = match *restart_count {
        1 => Duration::from_millis(250),
        2 => Duration::from_secs(1),
        _ => Duration::from_secs(4),
    };
    splash_server::diagnostics::print_error(&format!(
        "Engine failure; restarting attempt {}/{} in {:.2}s",
        restart_count,
        maximum,
        delay.as_secs_f64(),
    ));
    Ok(delay)
}

fn wait_restart_backoff(delay: Duration) -> bool {
    let deadline = Instant::now() + delay;
    while !STOP.load(Ordering::Relaxed) {
        let now = Instant::now();
        if now >= deadline {
            return true;
        }
        std::thread::sleep((deadline - now).min(Duration::from_millis(50)));
    }
    false
}

fn parse_thinking_budget(value: &str) -> Result<Option<u32>, Box<dyn Error>> {
    let budget = value.parse::<i64>()?;
    match budget {
        -1 => Ok(None),
        budget if (0..=u32::MAX as i64).contains(&budget) => Ok(Some(budget as u32)),
        _ => Err("thinking token budget must be non-negative or -1 for no cap".into()),
    }
}

fn parse_reasoning_effort(value: &str) -> Result<String, Box<dyn Error>> {
    match value {
        "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" => {
            Ok(value.to_owned())
        }
        _ => Err("--default-reasoning-effort must be none, minimal, low, medium, high, xhigh or max".into()),
    }
}

fn parse_expert_memory_grow(value: &str) -> Result<String, Box<dyn Error>> {
    match value {
        "on" | "off" => Ok(value.to_owned()),
        _ => Err("--expert-memory-grow must be on or off".into()),
    }
}

fn validate_announce_served_name(announce: bool, names: &[String]) -> Result<(), Box<dyn Error>> {
    if announce && names.is_empty() {
        Err("--announce-served-name needs --served-model-name".into())
    } else {
        Ok(())
    }
}

fn validate_served_model_name(value: &str) -> Result<(), Box<dyn Error>> {
    let invalid = value.is_empty()
        || value.chars().any(|character| {
            character.is_control()
                || character.is_whitespace()
                || matches!(character, '\\' | '%' | '?' | '#')
        })
        || value.split('/').any(|part| part.is_empty() || part == "." || part == "..");
    if invalid {
        return Err("model alias must be a non-empty name without whitespace or URL delimiters".into());
    }
    Ok(())
}

fn thinking_end_tokens(
    tokenizer: &FastTokenizer,
    text: Option<&str>,
    think_end_id: u32,
) -> Result<Option<Vec<u32>>, Box<dyn Error>> {
    let Some(text) = text else { return Ok(None) };
    let tokens = tokenizer.encode(text, false)?;
    let added: std::collections::HashSet<u32> =
        tokenizer.added_tokens().into_iter().map(|(id, _)| id).collect();
    if tokens.is_empty()
        || tokens.last().copied() != Some(think_end_id)
        || tokens[..tokens.len().saturating_sub(1)]
            .iter()
            .any(|token| added.contains(token))
    {
        return Err("--thinking-end-text must end with the chat format's thinking-end token and contain no other special token".into());
    }
    let maximum = splash_server::protocol::MAX_THINKING_END_TOKENS;
    if tokens.len() > maximum {
        return Err(format!("--thinking-end-text takes {} tokens; the most is {maximum}", tokens.len()).into());
    }
    Ok(Some(tokens))
}

fn engine_parity_args(
    expert_memory_grow: Option<&str>,
    disable_ane_mixer: bool,
    loop_guard: bool,
) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(value) = expert_memory_grow {
        args.extend(["--expert-memory-grow".into(), value.into()]);
    }
    if disable_ane_mixer {
        args.extend(["--ane-mixer".into(), "off".into()]);
    }
    if loop_guard {
        args.extend(["--loop-guard".into(), "on".into()]);
    }
    args
}

fn parse_compact_at(value: &str) -> Result<splash_server::serve::compaction::CompactAt, Box<dyn Error>> {
    if value == "auto" {
        return Ok(splash_server::serve::compaction::CompactAt::Auto);
    }
    let fraction = value.parse::<f64>()?;
    if !fraction.is_finite() || !(0.0 < fraction && fraction < 1.0) {
        return Err("--compact-at must be auto or a fraction between 0 and 1".into());
    }
    Ok(splash_server::serve::compaction::CompactAt::Fraction(fraction))
}

struct EngineProcess(Child);

impl Drop for EngineProcess {
    fn drop(&mut self) {
        if self.0.try_wait().is_ok_and(|status| status.is_some()) {
            return;
        }
        unsafe { libc::kill(self.0.id() as libc::pid_t, libc::SIGTERM); }
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(4) {
            if self.0.try_wait().is_ok_and(|status| status.is_some()) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn find_model(id: &str) -> Result<PathBuf, Box<dyn Error>> {
    let models = models_root()?;
    // A Hub model's selection link is the repository ID itself, or the
    // repository and variant the ID names (install/models.py's
    // selection_link); the link is what the installer prepared.
    let (repo, variant) = split_model_id(id);
    let link = match variant {
        Some(variant) => models.join(format!("{repo}:{variant}")),
        None => models.join(&repo),
    };
    if link.join("model.json").is_file() {
        return Ok(link.canonicalize()?);
    }
    find_assembly(|record| matches_model(record, id), id)
}

fn run_gguf_config(arguments: Vec<String>) -> Result<(), Box<dyn Error>> {
    let target_path = arguments.first().ok_or("gguf-config requires a GGUF file")?;
    let mut vision_path = None;
    let mut engine_path = None;
    let mut index = 1;
    while index < arguments.len() {
        let flag = &arguments[index];
        index += 1;
        let value = arguments.get(index).ok_or_else(|| format!("{flag} requires a path"))?.clone();
        index += 1;
        match flag.as_str() {
            "--vision" if vision_path.is_none() => vision_path = Some(PathBuf::from(value)),
            "--engine" if engine_path.is_none() => engine_path = Some(PathBuf::from(value)),
            "--vision" => return Err("gguf-config accepts --vision only once".into()),
            "--engine" => return Err("gguf-config accepts --engine only once".into()),
            _ => return Err(format!("unknown gguf-config option: {flag}").into()),
        }
    }

    let target = splash_server::gguf::Metadata::read(target_path, true)?;
    splash_server::gguf::require_single_file(&target)?;
    splash_server::gguf::require_loadable(&target)?;
    splash_server::gguf::validate_file_layout(Path::new(target_path), &target)?;
    let vision = vision_path.as_ref().map(|path| splash_server::gguf::Metadata::read(path, true)).transpose()?;
    if let Some(vision) = &vision {
        splash_server::gguf::require_single_file(vision)?;
        splash_server::gguf::require_vision_loadable(vision)?;
        splash_server::gguf::validate_file_layout(vision_path.as_ref().unwrap(), vision)?;
    }
    let config = splash_server::gguf::model_config(&target, vision.as_ref())?;
    let engine = resolve_check_engine(engine_path)?;
    let family = splash_server::gguf::validate_with_engine(&engine, &target, vision.as_ref())?;
    let result = serde_json::json!({
        "target_format": "gguf",
        "vision_format": if vision.is_some() { "gguf" } else { "none" },
        "family": family,
        "config": config,
        "gguf_metadata": target.scalar_metadata(),
        "validated_by": engine,
        "publish_ready": false,
        "publish_reason": "gguf-config is validation-only; use prepare-gguf with --models-dir to persist and publish the validated GGUF assembly",
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_prepare_gguf(arguments: Vec<String>) -> Result<(), Box<dyn Error>> {
    let target = arguments.first().ok_or("prepare-gguf requires a GGUF file")?;
    let mut models_dir = None;
    let mut vision = None;
    let mut engine = None;
    let mut model_id = None;
    let mut index = 1;
    while index < arguments.len() {
        let flag = &arguments[index];
        index += 1;
        let value = arguments.get(index).ok_or_else(|| format!("{flag} requires a value"))?.clone();
        index += 1;
        match flag.as_str() {
            "--models-dir" if models_dir.is_none() => models_dir = Some(PathBuf::from(value)),
            "--vision" if vision.is_none() => vision = Some(PathBuf::from(value)),
            "--engine" if engine.is_none() => engine = Some(PathBuf::from(value)),
            "--model-id" if model_id.is_none() => model_id = Some(value),
            "--models-dir" | "--vision" | "--engine" | "--model-id" => return Err(format!("prepare-gguf accepts {flag} only once").into()),
            _ => return Err(format!("unknown prepare-gguf option: {flag}").into()),
        }
    }
    let models_dir = models_dir.ok_or("prepare-gguf requires --models-dir DIR")?;
    let model_id = model_id.unwrap_or_else(|| {
        let stem = Path::new(target).file_stem().and_then(|name| name.to_str()).unwrap_or("local-gguf");
        let model_stem = stem.strip_suffix(".gguf").unwrap_or(stem);
        let (prefix, count) = model_stem.rsplit_once("-of-").unwrap_or((model_stem, ""));
        let (name, number) = prefix.rsplit_once('-').unwrap_or((prefix, ""));
        let is_split = number.len() == 5 && count.len() == 5
            && number.bytes().all(|byte| byte.is_ascii_digit())
            && count.bytes().all(|byte| byte.is_ascii_digit());
        format!("local/{}", if is_split { name } else { model_stem })
    });
    let engine = resolve_check_engine(engine)?;
    let assembly = splash_server::gguf::prepare_local(
        Path::new(target), vision.as_deref(), &models_dir, &model_id, &engine,
    )?;
    println!("{}", assembly.display());
    Ok(())
}

fn run_prepare_mlx(arguments: Vec<String>) -> Result<(), Box<dyn Error>> {
    let target = arguments.first().ok_or("prepare-mlx requires a model directory")?;
    let mut models_dir = None;
    let mut engine = None;
    let mut model_id = None;
    let mut language_only = false;
    let mut index = 1;
    while index < arguments.len() {
        let flag = &arguments[index];
        index += 1;
        if flag == "--language-only" {
            if language_only { return Err("prepare-mlx accepts --language-only only once".into()); }
            language_only = true;
            continue;
        }
        let value = arguments.get(index).ok_or_else(|| format!("{flag} requires a value"))?.clone();
        index += 1;
        match flag.as_str() {
            "--models-dir" if models_dir.is_none() => models_dir = Some(PathBuf::from(value)),
            "--engine" if engine.is_none() => engine = Some(PathBuf::from(value)),
            "--model-id" if model_id.is_none() => model_id = Some(value),
            "--models-dir" | "--engine" | "--model-id" => return Err(format!("prepare-mlx accepts {flag} only once").into()),
            _ => return Err(format!("unknown prepare-mlx option: {flag}").into()),
        }
    }
    let models_dir = models_dir.ok_or("prepare-mlx requires --models-dir DIR")?;
    let target_path = Path::new(target);
    let model_id = model_id.unwrap_or_else(|| {
        let name = target_path.file_name().and_then(|name| name.to_str()).unwrap_or("local-mlx");
        format!("local/{name}")
    });
    let engine = resolve_check_engine(engine)?;
    let assembly = splash_server::mlx_local::prepare_local(
        target_path, &models_dir, &model_id, &engine, language_only,
    )?;
    println!("{}", assembly.display());
    Ok(())
}

fn run_install_cached(arguments: Vec<String>) -> Result<(), Box<dyn Error>> {
    if arguments.as_slice() == ["--help"] || arguments.as_slice() == ["-h"] {
        println!("Usage: splash-server install-cached [--cache-dir DIR] [--models-dir DIR] [--engine PATH]\nScans local Hugging Face snapshots only; complete single/split GGUF is supported, while drafts and MLX models are skipped.");
        return Ok(());
    }
    let mut cache = None;
    let mut models = None;
    let mut engine = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        index += 1;
        let value = arguments.get(index).ok_or_else(|| format!("{flag} requires a path"))?.clone();
        index += 1;
        match flag.as_str() {
            "--cache-dir" if cache.is_none() => cache = Some(PathBuf::from(value)),
            "--models-dir" if models.is_none() => models = Some(PathBuf::from(value)),
            "--engine" if engine.is_none() => engine = Some(PathBuf::from(value)),
            "--cache-dir" | "--models-dir" | "--engine" => return Err(format!("install-cached accepts {flag} only once").into()),
            _ => return Err(format!("unknown install-cached option: {flag}").into()),
        }
    }
    let cache = cache.map(Ok).unwrap_or_else(splash_server::cached_install::default_hub_cache)?;
    let models = match models { Some(path) => path, None => models_root()? };
    let engine = resolve_check_engine(engine)?;
    for model in splash_server::cached_install::install_cached(&cache, &models, &engine)? {
        println!("{}", serde_json::json!({"model": model}));
    }
    Ok(())
}

fn resolve_check_engine(explicit: Option<PathBuf>) -> Result<PathBuf, Box<dyn Error>> {
    let mut candidates = Vec::new();
    if let Some(path) = explicit { candidates.push(path); }
    else {
        if let Some(path) = std::env::var_os("SPLASH_ENGINE_PATH") { candidates.push(PathBuf::from(path)); }
        if let Ok(executable) = std::env::current_exe() { candidates.push(executable.with_file_name("splash")); }
        if let Ok(directory) = std::env::current_dir() { candidates.push(directory.join("build/splash")); }
    }
    if let Some(path) = candidates.iter().find(|path| executable_file(path)) { return Ok(path.clone()); }
    let searched = candidates.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ");
    Err(format!("native engine for model-check is unavailable; pass --engine PATH (searched: {searched})").into())
}

fn executable_file(path: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    { path.is_file() }
}

#[cfg(test)]
mod cli_option_tests {
    use super::*;

    #[test]
    fn server_only_flags_do_not_enter_engine_arguments() {
        assert_eq!(
            engine_parity_args(Some("on"), true, true),
            vec![
                "--expert-memory-grow",
                "on",
                "--ane-mixer",
                "off",
                "--loop-guard",
                "on",
            ]
        );
        assert!(engine_parity_args(None, false, false).is_empty());
    }

    #[test]
    fn served_model_names_and_announcement_are_checked() {
        for name in ["owner/model", "local_alias", "team/model-v2"] {
            validate_served_model_name(name).expect("valid model name");
        }
        for name in ["", "a b", "a?b", "a#b", "a%2Fb", "a\\b", "a//b", "a/../b"] {
            assert!(validate_served_model_name(name).is_err(), "{name}");
        }
        assert!(validate_announce_served_name(true, &[]).is_err());
        assert!(validate_announce_served_name(true, &["alias".into()]).is_ok());
        assert!(validate_announce_served_name(false, &[]).is_ok());
    }

    #[test]
    fn reasoning_and_engine_switch_values_match_python_choices() {
        for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
            assert_eq!(parse_reasoning_effort(effort).unwrap(), effort);
        }
        assert!(parse_reasoning_effort("turbo").is_err());
        assert_eq!(parse_expert_memory_grow("on").unwrap(), "on");
        assert_eq!(parse_expert_memory_grow("off").unwrap(), "off");
        assert!(parse_expert_memory_grow("auto").is_err());
    }

    #[test]
    fn memory_and_cache_sizes_preserve_python_auto_off_and_zero_forms() {
        assert_eq!(parse_size_bytes("auto", true).unwrap(), None);
        assert_eq!(parse_size_bytes("2.5G", true).unwrap(), Some(2_684_354_560));
        assert_eq!(parse_memory_option("off", true).unwrap(), Some("off".into()));
        assert_eq!(parse_memory_option("auto", true).unwrap(), None);
        assert!(parse_memory_option("off", false).is_err());
        assert_eq!(parse_cache_disk("auto").unwrap(), None);
        assert_eq!(parse_cache_disk("0").unwrap(), Some("0".into()));
        assert_eq!(parse_cache_writes("0").unwrap(), 0);
        assert_eq!(parse_cache_writes("2G").unwrap(), 2 * 1024 * 1024 * 1024);
        assert!(parse_size_bytes(".5G", false).is_err());
        assert!(parse_size_bytes("1.G", false).is_err());
        assert!(parse_size_bytes("auto", false).is_err());
        assert!(parse_memory_option("off", false).is_err());
    }

    #[test]
    fn context_durations_and_numeric_engine_controls_are_bounded() {
        assert_eq!(parse_context("auto").unwrap(), "auto");
        assert_eq!(parse_context("100K").unwrap(), "102400");
        assert_eq!(parse_context("1M").unwrap(), "1000000");
        assert!(parse_context("1010001").is_err());
        assert!(parse_context("0").is_err());
        assert_eq!(parse_positive_duration("2h", "timeout").unwrap(), 7200.0);
        assert_eq!(parse_idle_duration("off", "idle").unwrap(), "off");
        assert!(parse_positive_duration("0s", "timeout").is_err());
        assert_eq!(parse_bounded_float("0", "decode", 0.0, None).unwrap(), "0");
        assert!(parse_bounded_float("NaN", "decode", 0.0, None).is_err());
        assert!(parse_bounded_float("10001", "slice", 1.0, Some(10_000.0)).is_err());
        assert!(parse_bounded_float("9", "duty", 10.0, Some(100.0)).is_err());
        assert_eq!(parse_ane_share("0").unwrap(), "0");
        assert!(parse_ane_share("1").is_err());
        assert!(parse_ane_share("NaN").is_err());
        assert!(validate_choice("int3", &["int8", "bf16"], "--kv-format").is_err());
        assert_eq!(parse_cache_eviction_protect("8").unwrap(), 8);
        assert_eq!(parse_cache_eviction_protect("0").unwrap(), 0);
        assert!(parse_cache_eviction_protect("9").is_err());
        assert!(parse_cache_eviction_protect("x").is_err());
        assert_eq!(parse_request_size("128M").unwrap(), 128 * 1024 * 1024);
        assert!(parse_request_size("0").is_err());
        assert_eq!(resolved_prefill_duty("balanced", None).as_deref(), Some("68"));
        assert_eq!(resolved_prefill_duty("quiet", Some("50".into())).as_deref(), Some("50"));
        assert_eq!(resolved_prefill_duty("auto", None), None);
        assert!(resolve_cache_dir("").is_err());
    }

    #[test]
    fn cache_dependencies_fail_before_engine_start() {
        assert!(validate_cache_dependencies(true, false, Some("0"), 0).is_err());
        assert!(validate_cache_dependencies(false, false, Some("0"), 1).is_err());
        assert!(validate_cache_dependencies(false, true, None, 0).is_err());
        assert!(validate_cache_dependencies(true, true, Some("1024"), 1).is_ok());
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Speculation {
    Auto,
    Dflash2,
}

impl Speculation {
    /// The mode a selection names, as `install/models.py`'s
    /// `speculation_mode`: `none` was one before this release, and is auto
    /// now.
    fn parse(value: &str) -> Result<Speculation, Box<dyn Error>> {
        match value {
            "auto" | "none" => Ok(Speculation::Auto),
            "dflash2" => Ok(Speculation::Dflash2),
            _ => Err(format!("unsupported speculation mode: {value}").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Speculation::Auto => "auto",
            Speculation::Dflash2 => "dflash2",
        }
    }
}

fn matches_model(record: &Value, id: &str) -> bool {
    record["model"] == id || record.pointer("/sources/target/repo").is_some_and(|repo| repo == id)
}

fn same_source(recorded: &str, requested: &str) -> bool {
    recorded == requested || matches!(
        (fs::canonicalize(recorded), fs::canonicalize(requested)),
        (Ok(recorded), Ok(requested)) if recorded == requested
    )
}

/// `owner/repo[:variant]` -> (repository ID, variant or None), as
/// `server/serve_options.py`'s `split_model_id`.
fn split_model_id(value: &str) -> (String, Option<String>) {
    match value.split_once(':') {
        Some((repo, variant)) if !variant.is_empty() => (repo.to_owned(), Some(variant.to_owned())),
        _ => (value.to_owned(), None),
    }
}

fn models_root() -> Result<PathBuf, Box<dyn Error>> {
    let home = std::env::var_os("HOME").ok_or("HOME is required to locate prepared models")?;
    Ok(PathBuf::from(home).join("Library/Application Support/Splash/models"))
}

/// The selection link a local target directory and its options name, as
/// `install/models.py`'s `local_selection_link` hashes it: the directory,
/// whether it serves text only, the draft, the speculation mode and whether
/// a draft is required, with the projector appended when one is named.
fn local_selection_link(
    models: &std::path::Path,
    directory: &str,
    language_only: bool,
    draft: Option<&str>,
    mmproj: Option<&str>,
    speculation: Speculation,
    require_draft: bool,
) -> PathBuf {
    let mut selection = vec![
        Value::String(directory.to_owned()),
        Value::Bool(language_only),
        draft.map_or(Value::Null, |draft| Value::String(draft.to_owned())),
        Value::String(speculation.name().to_owned()),
        Value::Bool(require_draft),
    ];
    if let Some(mmproj) = mmproj {
        selection.push(Value::String(mmproj.to_owned()));
    }
    let encoded = serde_json::to_string(&Value::Array(selection)).expect("selection JSON");
    models
        .join(".selections")
        .join(splash_server::support::sha256::hex_digest(encoded.as_bytes()))
}

/// The assembly a selection link names, when the installer prepared one.
fn assembly_at(link: &std::path::Path) -> Option<PathBuf> {
    link.join("model.json")
        .is_file()
        .then(|| link.canonicalize().ok())
        .flatten()
}

fn find_assembly(
    matches_record: impl Fn(&Value) -> bool,
    selection: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let models = models_root()?.join(".selections");
    let mut matches = Vec::new();
    for entry in fs::read_dir(&models)? {
        let path = entry?.path();
        if let Ok(bytes) = fs::read(path.join("model.json")) {
            if serde_json::from_slice::<Value>(&bytes).is_ok_and(|record| matches_record(&record)) {
                matches.push(path.canonicalize()?);
            }
        }
    }
    matches.sort();
    matches.dedup();
    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(format!("no prepared assembly for {selection}; select a prepared assembly directory").into()),
        _ => Err(format!("multiple prepared assemblies for {selection}; select the assembly directory").into()),
    }
}

fn bytes(value: &str) -> Result<u64, Box<dyn Error>> {
    parse_size_bytes(value, false)?.ok_or_else(|| "size cannot be auto".into())
}

fn parse_size_bytes(value: &str, allow_auto: bool) -> Result<Option<u64>, Box<dyn Error>> {
    let normalized = value.trim().to_ascii_uppercase();
    if allow_auto && normalized == "AUTO" {
        return Ok(None);
    }
    let mut number_end = normalized.len();
    let mut suffix = "";
    for candidate in ["KIB", "MIB", "GIB", "KB", "MB", "GB", "K", "M", "G", "B"] {
        if normalized.ends_with(candidate) {
            suffix = candidate;
            number_end -= candidate.len();
            break;
        }
    }
    let number = &normalized[..number_end];
    let mut parts = number.split('.');
    let whole = parts.next().unwrap_or_default();
    let fraction = parts.next();
    let valid_integer = !whole.is_empty() && whole.bytes().all(|byte| byte.is_ascii_digit());
    let valid_fraction = fraction.map_or(true, |part| {
        !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())
    });
    if !valid_integer || !valid_fraction || parts.next().is_some()
        || (fraction.is_some() && suffix.is_empty())
    {
        return Err("must be auto or a positive byte count such as 32G or 2.5G".into());
    }
    let multiplier = match suffix {
        "K" | "KB" | "KIB" => 1024u128,
        "M" | "MB" | "MIB" => 1024u128.pow(2),
        "G" | "GB" | "GIB" => 1024u128.pow(3),
        _ => 1,
    };
    let whole = whole.parse::<u128>()?;
    let mut total = whole.checked_mul(multiplier);
    if let Some(fraction) = fraction {
        let numerator = fraction.parse::<u128>()?;
        let denominator = 10u128.checked_pow(fraction.len() as u32);
        total = total
            .zip(denominator)
            .and_then(|(whole, denominator)| {
                numerator
                    .checked_mul(multiplier)
                    .and_then(|fraction| whole.checked_add(fraction / denominator))
            });
    }
    let Some(total) = total.filter(|total| *total > 0 && *total <= i64::MAX as u128) else {
        return Err("must be auto or a positive byte count such as 32G or 2.5G".into());
    };
    Ok(Some(total as u64))
}

fn parse_context(value: &str) -> Result<String, Box<dyn Error>> {
    let normalized = value.trim().to_ascii_uppercase();
    if normalized == "AUTO" {
        return Ok("auto".into());
    }
    let (digits, multiplier) = match normalized.as_bytes().last() {
        Some(b'K') => (&normalized[..normalized.len() - 1], 1024u64),
        Some(b'M') => (&normalized[..normalized.len() - 1], 1_000_000u64),
        _ => (normalized.as_str(), 1),
    };
    let count = digits.parse::<u64>().unwrap_or(0).saturating_mul(multiplier);
    if !(1..=1_010_000).contains(&count) {
        return Err("--max-context must be auto or a token count up to 1010000, such as 100K or 1M".into());
    }
    Ok(count.to_string())
}

fn parse_memory_option(value: &str, allow_off: bool) -> Result<Option<String>, Box<dyn Error>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "off" if allow_off => Ok(Some("off".into())),
        _ => Ok(parse_size_bytes(value, false)?.map(|size| size.to_string())),
    }
}

fn parse_cache_disk(value: &str) -> Result<Option<String>, Box<dyn Error>> {
    if value.trim() == "0" {
        return Ok(Some("0".into()));
    }
    parse_size_bytes(value, true).map(|size| size.map(|size| size.to_string()))
}

fn parse_cache_writes(value: &str) -> Result<u64, Box<dyn Error>> {
    if value.trim() == "0" {
        Ok(0)
    } else {
        bytes(value)
    }
}

fn parse_cache_eviction_protect(value: &str) -> Result<u32, Box<dyn Error>> {
    let count = value.trim().parse::<u32>().unwrap_or(u32::MAX);
    if count > 8 {
        Err("--cache-eviction-protect must be a whole number from 0 to 8".into())
    } else {
        Ok(count)
    }
}

fn parse_request_size(value: &str) -> Result<usize, Box<dyn Error>> {
    let size = bytes(value)?;
    if size == 0 || size > i64::MAX as u64 {
        return Err("--max-request-size must be a positive byte count".into());
    }
    Ok(usize::try_from(size)?)
}

fn validate_choice(value: &str, choices: &[&str], option: &str) -> Result<(), Box<dyn Error>> {
    if choices.contains(&value) {
        Ok(())
    } else {
        Err(format!("{option} must be one of {}", choices.join(", ")).into())
    }
}

fn resolved_prefill_duty(policy: &str, explicit: Option<String>) -> Option<String> {
    explicit.or_else(|| match policy {
        "balanced" => Some("68".into()),
        "moderate" => Some("48".into()),
        "quiet" => Some("33".into()),
        _ => None,
    })
}

fn parse_positive_duration(value: &str, option: &str) -> Result<f64, Box<dyn Error>> {
    let text = value.trim().to_ascii_lowercase();
    let (number, multiplier) = match text.as_bytes().last() {
        Some(b's') => (&text[..text.len() - 1], 1.0),
        Some(b'm') => (&text[..text.len() - 1], 60.0),
        Some(b'h') => (&text[..text.len() - 1], 3600.0),
        _ => (text.as_str(), 1.0),
    };
    let seconds = number.parse::<f64>().unwrap_or(f64::NAN) * multiplier;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(format!("{option} must be a positive duration such as 30m, 2h or 3600").into());
    }
    Ok(seconds)
}

fn parse_idle_duration(value: &str, option: &str) -> Result<String, Box<dyn Error>> {
    if value.trim().eq_ignore_ascii_case("off") {
        return Ok("off".into());
    }
    Ok(parse_positive_duration(value, option)?.to_string())
}

fn resolve_cache_dir(value: &str) -> Result<PathBuf, Box<dyn Error>> {
    if value.trim().is_empty() {
        return Err("--cache-dir must name a directory".into());
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(splash_server::cli_paths::account_home);
    let path = splash_server::cli_paths::expand_user_path(
        value,
        home.as_deref(),
        splash_server::cli_paths::named_account_home,
    );
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn parse_bounded_float(value: &str, option: &str, min: f64, max: Option<f64>) -> Result<String, Box<dyn Error>> {
    let parsed = value.parse::<f64>().unwrap_or(f64::NAN);
    if !parsed.is_finite() || parsed < min || max.is_some_and(|max| parsed > max) {
        return Err(match max {
            Some(max) => format!("{option} must be a finite number from {min} to {max}"),
            None => format!("{option} must be a finite number greater than or equal to {min}"),
        }.into());
    }
    Ok(parsed.to_string())
}

fn parse_ane_share(value: &str) -> Result<String, Box<dyn Error>> {
    let share = value.parse::<f64>().unwrap_or(f64::NAN);
    if !share.is_finite() || !(0.0..1.0).contains(&share) {
        return Err("SPLASH_ANE_SHARE must be a share from 0 to just below 1".into());
    }
    Ok(share.to_string())
}

fn validate_cache_dependencies(
    persistent: bool,
    cache_dir_set: bool,
    max_cache_disk: Option<&str>,
    max_cache_disk_writes: u64,
) -> Result<(), Box<dyn Error>> {
    if persistent && max_cache_disk == Some("0") {
        return Err("--persistent-cache needs the SSD cache; --max-cache-disk 0 turns it off".into());
    }
    if max_cache_disk_writes > 0 && max_cache_disk == Some("0") {
        return Err("--max-cache-disk-writes needs the SSD cache; --max-cache-disk 0 turns it off".into());
    }
    if cache_dir_set && !persistent {
        return Err("--cache-dir needs --persistent-cache".into());
    }
    Ok(())
}
