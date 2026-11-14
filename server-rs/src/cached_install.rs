//! Offline installation of single-file GGUFs already present in a Hub cache.
//!
//! This is deliberately read-only with respect to cached snapshots and blobs.
//! The only cache write is a Hub reference pin after a verified assembly has
//! been published, so external cache pruning cannot invalidate the selection.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn default_hub_cache() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("HF_HUB_CACHE") {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        return Ok(PathBuf::from(path).join("hub"));
    }
    let home = std::env::var_os("HOME").ok_or("HOME is required to locate the Hugging Face cache")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Install supported single-file GGUFs from the local `models--*` snapshots.
/// No Hub client or network path is involved. Unsupported model repositories
/// are skipped with a concise stderr explanation.
pub fn install_cached(
    hub_cache: &Path,
    models_root: &Path,
    engine: &Path,
) -> Result<Vec<String>, String> {
    let mut installed = Vec::new();
    let repositories = match fs::read_dir(hub_cache) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(installed),
        Err(error) => return Err(format!("cannot scan Hugging Face cache {}: {error}", hub_cache.display())),
    };
    let mut directories = repositories
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("models--"))
        .collect::<Vec<_>>();
    directories.sort_by_key(|entry| entry.file_name());

    let suggestions = suggested_models();
    for entry in directories {
        let folder_name = entry.file_name().to_string_lossy().into_owned();
        let repository = folder_name.trim_start_matches("models--").replace("--", "/");
        if !valid_repository_id(&repository) {
            continue;
        }
        let refs = entry.path().join("refs/main");
        let commit = match fs::read_to_string(&refs) {
            Ok(commit) if is_hex_digest(commit.trim(), 40) => commit.trim().to_ascii_lowercase(),
            _ => continue,
        };
        let snapshot = entry.path().join("snapshots").join(&commit);
        if !snapshot.is_dir() { continue; }
        let files = match snapshot_files(&snapshot) {
            Ok(files) => files,
            Err(error) => {
                eprintln!("skip {repository}: cannot read cached snapshot: {error}");
                continue;
            }
        };
        let all_gguf = files.keys().filter(|name| name.ends_with(".gguf")).collect::<Vec<_>>();
        let root_gguf = all_gguf.iter().filter(|name| !name.contains('/')).copied().collect::<Vec<_>>();
        let targets = match cached_targets(&root_gguf) {
            Ok(targets) => targets,
            Err(error) => {
                eprintln!("skip {repository}: {error}");
                continue;
            }
        };
        if targets.is_empty() {
            if !all_gguf.is_empty() {
                eprintln!("skip {repository}: split or projector-only GGUF is unsupported");
            } else if files.keys().any(|name| name.ends_with(".safetensors")) {
                eprintln!("skip {repository}: MLX/draft repositories are not supported by install-cached");
            }
            continue;
        }
        if targets.iter().any(|target| {
            Path::new(&target.model).file_stem().and_then(|stem| stem.to_str())
                .is_some_and(|stem| { let stem = stem.to_ascii_lowercase(); stem.contains("dflash") || stem.contains("draft") })
        }) {
            eprintln!("skip {repository}: draft GGUF repositories are unsupported");
            continue;
        }
        if all_gguf.len() != root_gguf.len()
        {
            eprintln!("skip {repository}: nested GGUF files are unsupported");
            continue;
        }
        let target_ids = candidate_ids(&repository, &targets, &suggestions);
        let projectors = root_gguf.iter().filter(|name| projector_named(name)).collect::<Vec<_>>();
        let vision = match projectors.as_slice() {
            [] => None,
            [one] => files.get(one.as_str()).map(PathBuf::as_path),
            _ => {
                eprintln!("skip {repository}: multiple cached vision projectors need explicit selection");
                continue;
            }
        };

        for (target_name, model_id) in target_ids {
            let Some(target_path) = files.get(&target_name) else { continue; };
            let link = models_root.join(&model_id);
            if fs::symlink_metadata(&link).is_ok() { continue; }
            let assembly = match crate::gguf::prepare_local(target_path, vision, models_root, &model_id, engine) {
                Ok(path) => path,
                Err(error) => {
                    eprintln!("skip {model_id}: {error}");
                    continue;
                }
            };
            if let Err(error) = crate::assembly::publish_selection(&link, &assembly) {
                eprintln!("skip {model_id}: cannot publish selection: {error}");
                continue;
            }
            if let Err(error) = write_hub_pin(&entry.path(), &snapshot, &link, &commit) {
                // The published model remains valid, but a cache manager could
                // later remove its source blob. Match Python's warning-only
                // behavior when a cache cannot retain a reference.
                eprintln!("warning: installed {model_id}, but could not pin its Hub snapshot: {error}");
            }
            installed.push(model_id);
        }
    }
    if let Err(error) = crate::assembly::collect_generated(models_root) {
        eprintln!("warning: generated source cache cleanup failed: {error}");
    }
    Ok(installed)
}

fn snapshot_files(root: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    fn visit(root: &Path, current: &Path, output: &mut BTreeMap<String, PathBuf>) -> Result<(), String> {
        for entry in fs::read_dir(current).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            if kind.is_dir() {
                visit(root, &path, output)?;
            } else if kind.is_file() || (kind.is_symlink() && fs::metadata(&path).is_ok_and(|m| m.is_file())) {
                let relative = path.strip_prefix(root).map_err(|error| error.to_string())?
                    .to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
                if safe_relative(&relative) { output.insert(relative, path); }
            }
        }
        Ok(())
    }
    let mut output = BTreeMap::new();
    visit(root, root, &mut output)?;
    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedTarget { representative: String, model: String }

fn cached_targets(root_gguf: &[&String]) -> Result<Vec<CachedTarget>, String> {
    let mut singles = Vec::new();
    let mut splits = BTreeMap::<String, BTreeMap<u32, (u32, String)>>::new();
    for name in root_gguf.iter().filter(|name| !projector_named(name)) {
        if let Some((model, number, count)) = split_filename(name) {
            let group = splits.entry(format!("{model}.gguf")).or_default();
            if group.insert(number, (count, (*name).clone())).is_some() {
                return Err(format!("duplicate split GGUF part: {name}"));
            }
        } else {
            singles.push(CachedTarget { representative: (*name).clone(), model: (*name).clone() });
        }
    }
    for (model, parts) in splits {
        let expected_count = parts.values().next().map(|(count, _)| *count).unwrap_or(0);
        if parts.values().any(|(count, _)| *count != expected_count)
            || parts.keys().copied().collect::<Vec<_>>() != (1..=expected_count).collect::<Vec<_>>()
        {
            return Err(format!("split GGUF {model} is incomplete or has inconsistent part numbers"));
        }
        let representative = parts.get(&1).map(|(_, name)| name.clone()).ok_or_else(|| format!("split GGUF {model} is missing part 1"))?;
        singles.push(CachedTarget { representative, model });
    }
    singles.sort_by(|a, b| a.model.cmp(&b.model));
    Ok(singles)
}

fn candidate_ids(repo: &str, targets: &[CachedTarget], suggestions: &[String]) -> Vec<(String, String)> {
    let mut ordered_targets = targets.to_vec();
    ordered_targets.sort_by(|a, b| a.model.cmp(&b.model));
    let all = ordered_targets.iter().map(|target| target.model.clone()).collect::<Vec<_>>();
    let mut result = Vec::new();
    for target in &ordered_targets {
        let stem = Path::new(&target.model).file_stem().and_then(|name| name.to_str()).unwrap_or("");
        let repo_words = repo.rsplit('/').next().unwrap_or(repo).to_ascii_lowercase()
            .strip_suffix("-gguf").unwrap_or(repo.rsplit('/').next().unwrap_or(repo)).split('-')
            .map(str::to_owned).collect::<Vec<_>>();
        let target_words = stem.split('-').map(|word| word.to_ascii_lowercase()).collect::<Vec<_>>();
        let shared = common_prefix(&repo_words, &target_words);
        let variant = target_words.iter().skip(shared.max(1)).cloned().collect::<Vec<_>>().join("-");
        let own_id = if variant.is_empty() { repo.to_owned() } else { format!("{repo}:{variant}") };
        let candidates = suggestions.iter().filter(|id| id.split_once(':').is_some_and(|(owner, _)| owner == repo))
            .chain(std::iter::once(&own_id)).cloned().collect::<Vec<_>>();
        let chosen = candidates.into_iter().find(|id| selects_target(repo, id, &target.model, &all)).unwrap_or(own_id);
        result.push((target.representative.clone(), chosen));
    }
    result
}

fn selects_target(repo: &str, model_id: &str, target: &str, all: &[String]) -> bool {
    let variant = model_id.strip_prefix(repo).and_then(|suffix| suffix.strip_prefix(':'));
    if variant.is_none() { return all.len() == 1 && all[0] == target; }
    let variant = variant.unwrap().to_ascii_lowercase();
    let stems = all.iter().map(|name| Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("")
        .split('-').map(|word| word.to_ascii_lowercase()).collect::<Vec<_>>()).collect::<Vec<_>>();
    let shared = stems.iter().skip(1).fold(stems.first().cloned().unwrap_or_default(), |prefix, words| {
        prefix.into_iter().zip(words).take_while(|(a, b)| a == *b).map(|(a, _)| a).collect()
    }).len();
    let exact = all.iter().zip(&stems).filter(|(_, words)| words.iter().skip(shared).cloned().collect::<Vec<_>>().join("-") == variant)
        .map(|(name, _)| name.as_str()).collect::<Vec<_>>();
    if exact.len() == 1 { return exact[0] == target; }
    let ending = all.iter().filter(|name| Path::new(name).file_stem().and_then(|s| s.to_str())
        .is_some_and(|stem| stem.to_ascii_lowercase().ends_with(&format!("-{variant}"))))
        .map(String::as_str).collect::<Vec<_>>();
    ending.len() == 1 && ending[0] == target
}

fn suggested_models() -> Vec<String> {
    include_str!("../../install/suggested-models.tsv").lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| line.split('\t').next().map(str::to_owned)).collect()
}

fn write_hub_pin(repo_folder: &Path, snapshot: &Path, selection: &Path, commit: &str) -> Result<(), String> {
    if !is_hex_digest(commit, 40)
        || snapshot.file_name().and_then(|name| name.to_str()) != Some(commit)
        || snapshot.parent().and_then(Path::file_name).and_then(|name| name.to_str()) != Some("snapshots")
        || snapshot.parent().and_then(Path::parent) != Some(repo_folder)
    {
        return Err("snapshot does not match its cache repository and commit".into());
    }
    let selection_parent = selection.parent().ok_or("invalid selection path")?.canonicalize().map_err(|error| error.to_string())?;
    let selection_name = selection.file_name().ok_or("invalid selection path")?;
    let owner = selection_parent.join(selection_name);
    let owner_hash = crate::support::sha256::hex_digest(owner.to_string_lossy().as_bytes());
    let reference = repo_folder.join("refs/splash").join(owner_hash).join(commit);
    match fs::read_to_string(&reference) {
        Ok(value) if value == commit => return Ok(()),
        Ok(_) => return Err(format!("invalid installed snapshot reference: {}", reference.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("cannot read Hub cache reference {}: {error}", reference.display())),
    }
    let parent = reference.parent().ok_or("invalid Hub pin path")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    for attempt in 0..32 {
        let temporary = parent.join(format!(".pending-{}-{attempt}", std::process::id()));
        let mut file = match fs::OpenOptions::new().write(true).create_new(true).open(&temporary) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.to_string()),
        };
        if let Err(error) = file.write_all(commit.as_bytes()).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&temporary);
            return Err(error.to_string());
        }
        match fs::rename(&temporary, &reference) {
            Ok(()) => return Ok(()),
            Err(error) => { let _ = fs::remove_file(&temporary); return Err(error.to_string()); }
        }
    }
    Err("could not allocate a temporary Hub reference file".into())
}

fn valid_repository_id(value: &str) -> bool {
    let Some((owner, repo)) = value.split_once('/') else { return false; };
    !owner.is_empty() && !repo.is_empty() && !owner.contains('/') && !repo.contains('/')
        && owner != "." && owner != ".." && repo != "." && repo != ".."
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b))
}
fn safe_relative(value: &str) -> bool {
    !value.is_empty() && !value.starts_with('/') && !value.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        && !value.chars().any(char::is_control)
}
fn projector_named(name: &str) -> bool { Path::new(name).file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.to_ascii_lowercase().contains("mmproj")) }
fn split_filename(name: &str) -> Option<(&str, u32, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let (prefix, count) = stem.rsplit_once("-of-")?;
    let (model, number) = prefix.rsplit_once('-')?;
    if number.len() != 5 || count.len() != 5
        || !number.bytes().all(|byte| byte.is_ascii_digit())
        || !count.bytes().all(|byte| byte.is_ascii_digit()) { return None; }
    Some((model, number.parse().ok()?, count.parse().ok()?))
}
fn common_prefix(a: &[String], b: &[String]) -> usize { a.iter().zip(b).take_while(|(a, b)| a == b).count() }
fn is_hex_digest(value: &str, length: usize) -> bool { value.len() == length && value.bytes().all(|b| b.is_ascii_hexdigit()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_repo_candidates_preserve_suggested_variant_selection() {
        let targets = vec![
            CachedTarget { representative: "Qwen3.8-27B-UD-Q4_K_M.gguf".to_owned(), model: "Qwen3.8-27B-UD-Q4_K_M.gguf".to_owned() },
            CachedTarget { representative: "Qwen3.8-27B-UD-IQ3_XXS.gguf".to_owned(), model: "Qwen3.8-27B-UD-IQ3_XXS.gguf".to_owned() },
        ];
        let suggestions = vec![
            "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M".to_owned(),
            "unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS".to_owned(),
        ];
        assert_eq!(candidate_ids("unsloth/Qwen3.8-27B-GGUF", &targets, &suggestions), vec![
            (targets[1].representative.clone(), "unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS".into()),
            (targets[0].representative.clone(), "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M".into()),
        ]);
    }

    #[test]
    fn cached_split_targets_are_grouped_by_model_and_require_complete_numbering() {
        let names = ["model-UD-Q4_K_M-00001-of-00002.gguf", "model-UD-Q4_K_M-00002-of-00002.gguf"];
        let refs = names.iter().map(|name| (*name).to_owned()).collect::<Vec<_>>();
        let target_refs = refs.iter().collect::<Vec<_>>();
        let targets = cached_targets(&target_refs).unwrap();
        assert_eq!(targets, vec![CachedTarget {
            representative: names[0].to_owned(), model: "model-UD-Q4_K_M.gguf".to_owned()
        }]);
        let suggestions = vec!["owner/model:UD-Q4_K_M".to_owned()];
        assert_eq!(candidate_ids("owner/model", &targets, &suggestions), vec![
            (names[0].to_owned(), "owner/model:UD-Q4_K_M".to_owned())
        ]);
        assert!(cached_targets(&[&refs[0]]).unwrap_err().contains("incomplete"));
    }

    #[test]
    fn snapshot_scanner_reads_symlinked_snapshot_files_without_touching_blobs() {
        let root = std::env::temp_dir().join(format!("splash-hub-snapshot-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("snapshot")).unwrap();
        fs::create_dir_all(root.join("blobs")).unwrap();
        fs::write(root.join("blobs/opaque"), b"gguf").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("blobs/opaque"), root.join("snapshot/model.gguf")).unwrap();
        let files = snapshot_files(&root.join("snapshot")).unwrap();
        assert!(files.contains_key("model.gguf"));
        assert_eq!(fs::read(root.join("blobs/opaque")).unwrap(), b"gguf");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn hub_snapshot_pin_is_scoped_to_the_selection_without_removing_other_refs() {
        let root = std::env::temp_dir().join(format!("splash-hub-pin-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let repo = root.join("models--owner--repo");
        let commit = "a".repeat(40);
        let snapshot = repo.join("snapshots").join(&commit);
        fs::create_dir_all(&snapshot).unwrap();
        let models = root.join("models");
        let selection = models.join("owner/repo");
        fs::create_dir_all(selection.parent().unwrap()).unwrap();
        let existing = repo.join("refs/splash/other/").join("b".repeat(40));
        fs::create_dir_all(existing.parent().unwrap()).unwrap();
        fs::write(&existing, "b".repeat(40)).unwrap();

        write_hub_pin(&repo, &snapshot, &selection, &commit).unwrap();
        let owner = selection.parent().unwrap().canonicalize().unwrap().join("repo");
        let owner_hash = crate::support::sha256::hex_digest(owner.to_string_lossy().as_bytes());
        let pin = repo.join("refs/splash").join(owner_hash).join(&commit);
        assert_eq!(fs::read_to_string(&pin).unwrap(), commit);
        assert_eq!(fs::read_to_string(existing).unwrap(), "b".repeat(40));
        fs::write(&pin, "wrong-commit").unwrap();
        assert!(write_hub_pin(&repo, &snapshot, &selection, &commit).unwrap_err().contains("invalid installed snapshot reference"));
        assert_eq!(fs::read_to_string(&pin).unwrap(), "wrong-commit", "an existing mismatched ref is never overwritten");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repository_ids_cannot_escape_model_selection_paths() {
        assert!(valid_repository_id("unsloth/model-Q4"));
        for invalid in ["../repo", "owner/..", "./repo", "owner/.", "owner/repo/extra", "/repo", "owner/"] {
            assert!(!valid_repository_id(invalid), "accepted unsafe repository ID: {invalid}");
        }
    }
}
