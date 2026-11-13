use std::error::Error;
use std::fs::{self, File};
use std::net::{IpAddr, TcpListener};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use splash_server::chat_format::{flint, qwen};
use splash_server::engine::EngineClient;
use splash_server::serve::{Pipeline, Server};
use splash_server::templates::load_chat_templates;
use splash_server::tokenization::{FastTokenizer, PromptTokenizer};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn main() {
    if let Err(error) = run() {
        eprintln!("splash-server: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("--help" | "-h") => {
            println!("Usage: splash-server serve (--model ID | --model-dir PATH) [--draft-model PATH_OR_ID] [--port PORT] [--binary ENGINE]");
            println!("       splash-server model-path ID");
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
    let mut max_image_pixels = splash_server::images::MAX_PIXELS;
    let mut binary = None;
    let mut pending = 8;
    let mut context = "auto".to_owned();
    let mut memory = "auto".to_owned();
    let mut disk = None;
    let mut engine_args = Vec::new();
    let mut key = std::env::var("SPLASH_API_KEY").ok().filter(|key| !key.is_empty());
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--require-draft" => require_draft = true,
            "--language-only" => language_only = true,
            "--disable-ane" => engine_args.extend(["--ane".into(), "off".into()]),
            "--persistent-cache" => engine_args.extend([flag, "on".into()]),
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
                    "--port" => port = value.parse()?,
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
                    "--host" if value.parse::<IpAddr>()?.is_loopback() => {
                        if value != "127.0.0.1" {
                            return Err("this build binds to 127.0.0.1".into());
                        }
                    }
                    "--host" => return Err("LAN access requires the full HTTP listener port".into()),
                    "--max-context" => context = value,
                    "--max-memory" => memory = bytes(&value)?.to_string(),
                    "--max-cache-disk" => disk = Some(bytes(&value)?.to_string()),
                    "--max-cache-disk-writes" | "--ple-memory" | "--expert-memory"
                    | "--sparse-kv-memory" => engine_args.extend([flag, bytes(&value)?.to_string()]),
                    "--kv-format" | "--state-format" | "--idle-release" | "--residency-keepalive"
                    | "--cache-idle-spill" | "--cache-idle-evict" | "--cache-dir"
                    | "--cache-eviction" | "--cache-eviction-protect" | "--verify-rows"
                    | "--decode-share" | "--prefill-slice-ms" | "--prefill-duty" => {
                        engine_args.extend([flag, value]);
                    }
                    "--thermal-policy" => {
                        let duty = match value.as_str() {
                            "auto" => "100",
                            "balanced" => "68",
                            "moderate" => "48",
                            "quiet" => "33",
                            _ => return Err("unknown thermal policy".into()),
                        };
                        engine_args.extend(["--prefill-duty".into(), duty.into()]);
                    }
                    _ => return Err(format!("option {flag} is not implemented in the Rust server").into()),
                }
            }
        }
    }
    if pending == 0 {
        return Err("--queue-size must be positive".into());
    }
    if let Some(key) = &key {
        splash_server::http::validate_api_key(key)?;
    }
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
            let link = local_selection_link(
                &models,
                &target,
                language_only,
                draft.as_deref(),
                mmproj.as_deref(),
                speculation,
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
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let tokenizer_directory = path.join("tokenizer");
    let tokenizer = Arc::new(FastTokenizer::from_file(&tokenizer_directory)?);
    let templates = load_chat_templates(&tokenizer_directory)?;
    let prompt_tokenizer = PromptTokenizer::from_directory(
        tokenizer.clone(), &tokenizer_directory, format.message_boundary,
    );
    // The compiled output constraints, over the model's grammar tokenizer:
    // constrained tool calls and structured output generate under them.
    let constraints = splash_server::constraints::ConstraintTokenizer::from_directory(
        &tokenizer_directory,
        format.vocabulary,
        &format.stop_ids(),
    )
    .ok()
    .and_then(|env| splash_server::constraints::ConstraintFactory::new(&env, &format).ok());
    let binary = match binary {
        Some(binary) => binary,
        None => std::env::current_exe()?.with_file_name("splash"),
    };
    let mut command = Command::new(binary);
    command.arg("serve-native").arg(path).args([context, memory]).args(disk).args(engine_args);
    command.stdin(Stdio::piped()).stdout(Stdio::piped());

    unsafe {
        libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
    }
    let mut child = EngineProcess(command.spawn()?);
    let client = EngineClient::over_streams(
        Box::new(child.0.stdout.take().expect("piped stdout")),
        Box::new(child.0.stdin.take().expect("piped stdin")),
        pending,
    );
    let start = Instant::now();
    while !client.wait_ready(Some(Duration::from_millis(100))) {
        if STOP.load(Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(error) = client.fatal_error() {
            return Err(error.into());
        }
        if let Some(status) = child.0.try_wait()? {
            return Err(format!("engine exited during startup: {status}").into());
        }
        if start.elapsed() >= Duration::from_secs(180) {
            return Err("engine did not become ready within 180 seconds".into());
        }
    }
    eprintln!("Ready · {model} · http://{}", listener.local_addr()?);
    let mut engine_exited = None;
    Server::new(Pipeline {
        tokenizer, prompt_tokenizer, templates, chat_format: format, client, model,
        images: splash_server::images::ImageCache::new(),
        max_image_pixels,
        constraints,
    }).with_api_key(key).serve_until(listener, || {
        if STOP.load(Ordering::Relaxed) {
            return true;
        }
        engine_exited = child.0.try_wait().transpose();
        engine_exited.is_some()
    })?;
    if let Some(status) = engine_exited {
        return Err(format!("engine exited while serving: {}", status?).into());
    }
    Ok(())
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
    let value = value.trim().to_ascii_uppercase();
    let value = value.strip_suffix('B').unwrap_or(&value);
    let (number, scale) = match value.strip_suffix('I').unwrap_or(value).as_bytes().last() {
        Some(b'K') => (1, 1024_u64),
        Some(b'M') => (1, 1024_u64.pow(2)),
        Some(b'G') => (1, 1024_u64.pow(3)),
        Some(b'T') => (1, 1024_u64.pow(4)),
        _ => (0, 1),
    };
    let value = value.strip_suffix('I').unwrap_or(value);
    let quantity: f64 = value[..value.len() - number].parse()?;
    let total = quantity * scale as f64;
    if !total.is_finite() || total < 0.0 || total >= u64::MAX as f64 {
        return Err("size is outside the supported byte range".into());
    }
    Ok(total as u64)
}
