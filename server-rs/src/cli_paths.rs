//! Shell-style home expansion for CLI paths.

use std::path::{Path, PathBuf};

/// Expand a leading `~` using the current user's home and `~name` using the
/// named user's home. Unknown users are left unchanged, matching Python's
/// `os.path.expanduser` behavior.
pub fn expand_user_path(
    value: &str,
    current_home: Option<&Path>,
    mut named_home: impl FnMut(&str) -> Option<PathBuf>,
) -> PathBuf {
    let Some(rest) = value.strip_prefix('~') else {
        return PathBuf::from(value);
    };

    if rest.is_empty() || rest.starts_with('/') {
        return current_home
            .map(|home| home.join(rest.strip_prefix('/').unwrap_or(rest)))
            .unwrap_or_else(|| PathBuf::from(value));
    }

    let boundary = rest.find('/').unwrap_or(rest.len());
    let user = &rest[..boundary];
    let suffix = &rest[boundary..];
    named_home(user)
        .map(|home| home.join(suffix.strip_prefix('/').unwrap_or(suffix)))
        .unwrap_or_else(|| PathBuf::from(value))
}

/// Current user's home directory from the account database. The caller can
/// prefer `$HOME`, as Python does, and use this as a fallback.
#[cfg(unix)]
pub fn account_home() -> Option<PathBuf> {
    use std::ffi::CStr;
    use std::os::unix::ffi::OsStrExt;

    // This is only called during single-threaded CLI parsing. Copy the
    // account database's static buffer before returning to avoid retaining a
    // pointer across another libc lookup.
    let entry = unsafe { libc::getpwuid(libc::getuid()) };
    if entry.is_null() {
        return None;
    }
    let home = unsafe { (*entry).pw_dir };
    if home.is_null() {
        return None;
    }
    let bytes = unsafe { CStr::from_ptr(home) }.to_bytes();
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

#[cfg(not(unix))]
pub fn account_home() -> Option<PathBuf> {
    None
}

/// Look up a named account's home directory, copying it from libc's static
/// account database buffer before returning.
#[cfg(unix)]
pub fn named_account_home(user: &str) -> Option<PathBuf> {
    use std::ffi::{CStr, CString};
    use std::os::unix::ffi::OsStrExt;

    let user = CString::new(user).ok()?;
    let entry = unsafe { libc::getpwnam(user.as_ptr()) };
    if entry.is_null() {
        return None;
    }
    let home = unsafe { (*entry).pw_dir };
    if home.is_null() {
        return None;
    }
    let bytes = unsafe { CStr::from_ptr(home) }.to_bytes();
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

#[cfg(not(unix))]
pub fn named_account_home(_user: &str) -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_current_and_named_homes_only_at_the_start() {
        let home = Path::new("/home/current");
        let named = |user: &str| (user == "other").then(|| PathBuf::from("/home/other"));

        assert_eq!(expand_user_path("~", Some(home), named), home);
        assert_eq!(expand_user_path("~/cache", Some(home), named), home.join("cache"));
        assert_eq!(expand_user_path("~other/cache", Some(home), named), PathBuf::from("/home/other/cache"));
        assert_eq!(expand_user_path("x/~other", Some(home), named), PathBuf::from("x/~other"));
    }

    #[test]
    fn unknown_named_users_and_missing_home_are_left_unchanged() {
        assert_eq!(expand_user_path("~missing/cache", None, |_| None), PathBuf::from("~missing/cache"));
        assert_eq!(expand_user_path("~/cache", None, |_| None), PathBuf::from("~/cache"));
    }
}
