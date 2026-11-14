use std::path::PathBuf;
use std::process;

use splash_build_tools::{check_identity, parse_constants, stale_outputs, write_identity, ToolError};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), ToolError> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("build-config") => run_build_config(args.collect()),
        Some("build-identity") => run_build_identity(args.collect()),
        _ => Err(ToolError("usage: splash-build-tools {build-config|build-identity} ...".into())),
    }
}

fn run_build_config(args: Vec<String>) -> Result<(), ToolError> {
    match args.first().map(String::as_str) {
        Some("stale") => {
            let mut index = 1;
            let mut config = None;
            let mut outputs = Vec::new();
            while index < args.len() {
                match args[index].as_str() {
                    "--config" => { index += 1; config = Some(args.get(index).ok_or_else(|| ToolError("stale requires --config VALUE".into()))?.clone()); }
                    value if value.starts_with('-') => return Err(ToolError(format!("unknown stale option: {value}"))),
                    output => outputs.push(output.to_owned()),
                }
                index += 1;
            }
            let config = config.ok_or_else(|| ToolError("stale requires --config VALUE".into()))?;
            println!("{}", stale_outputs(&config, &outputs).join(" "));
            Ok(())
        }
        Some("record") => {
            let mut index = 1;
            let mut config = None;
            let mut output: Option<PathBuf> = None;
            let mut command = Vec::new();
            while index < args.len() {
                match args[index].as_str() {
                    "--config" => { index += 1; config = Some(args.get(index).ok_or_else(|| ToolError("record requires --config VALUE".into()))?.clone()); }
                    "--output" => { index += 1; output = Some(PathBuf::from(args.get(index).ok_or_else(|| ToolError("record requires --output PATH".into()))?)); }
                    "--" => { command.extend_from_slice(&args[index + 1..]); break; }
                    value if value.starts_with('-') => return Err(ToolError(format!("unknown record option: {value}"))),
                    value => return Err(ToolError(format!("unexpected record argument before --: {value}"))),
                }
                index += 1;
            }
            let output = output.ok_or_else(|| ToolError("record requires --output PATH".into()))?;
            let config = config.ok_or_else(|| ToolError("record requires --config VALUE".into()))?;
            if command.is_empty() { return Err(ToolError("record requires a compiler or linker command after --".into())); }
            process::exit(splash_build_tools::run_build_config_record(&output, &config, &command).map_err(|error| ToolError(format!("build-config: {error}")))?);
        }
        _ => Err(ToolError("usage: splash-build-tools build-config {stale|record} ...".into())),
    }
}

fn run_build_identity(args: Vec<String>) -> Result<(), ToolError> {
    let mode = args.first().map(String::as_str).ok_or_else(|| ToolError("usage: splash-build-tools build-identity {write|check} ...".into()))?;
    if mode != "write" && mode != "check" { return Err(ToolError(format!("unknown build-identity mode: {mode}"))); }
    let mut root = splash_build_tools::default_root();
    let mut header = None;
    let mut stamp = None;
    let mut constants = Vec::new();
    let mut binaries = Vec::new();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--root" => { index += 1; root = PathBuf::from(args.get(index).ok_or_else(|| ToolError("--root requires a path".into()))?); }
            "--header" => { index += 1; header = Some(PathBuf::from(args.get(index).ok_or_else(|| ToolError("--header requires a path".into()))?)); }
            "--stamp" => { index += 1; stamp = Some(PathBuf::from(args.get(index).ok_or_else(|| ToolError("--stamp requires a path".into()))?)); }
            "--constant" => { index += 1; constants.push(args.get(index).ok_or_else(|| ToolError("--constant requires NAME=VALUE".into()))?.clone()); }
            "--binary" => { index += 1; binaries.push(PathBuf::from(args.get(index).ok_or_else(|| ToolError("--binary requires a path".into()))?)); }
            flag => return Err(ToolError(format!("unknown build-identity option: {flag}"))),
        }
        index += 1;
    }
    let header = header.ok_or_else(|| ToolError("--header is required".into()))?;
    let stamp = stamp.ok_or_else(|| ToolError("--stamp is required".into()))?;
    let constants = parse_constants(&constants)?;
    if mode == "write" {
        let (identity, header_changed, stamp_changed) = write_identity(&root, &constants, &header, &stamp)?;
        println!("{{\"build_id\":\"{identity}\",\"header_changed\":{header_changed},\"stamp_changed\":{stamp_changed}}}");
    } else {
        let identity = check_identity(&root, &constants, &header, &stamp, &binaries)?;
        println!("{{\"build_id\":\"{identity}\",\"verified\":true}}");
    }
    Ok(())
}
