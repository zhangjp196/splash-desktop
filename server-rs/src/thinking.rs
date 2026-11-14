//! Fernet-backed signatures for client-carried hidden reasoning.
//!
//! The token layout and key format match Python's `cryptography.fernet.Fernet`.
//! Cryptographic primitives use macOS CommonCrypto, which is available on the
//! platform this server targets; no Rust crypto crate is currently in the lockfile.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::URL_SAFE;
use base64::Engine;

use crate::support::errors::ApiError;

const KEY_BYTES: usize = 32;
const KEY_ENCODED_BYTES: usize = 44;
pub const MAX_SIGNATURE_BYTES: usize = 16 * 1024 * 1024;
const INVALID_SIGNATURE: &str = "invalid signature in thinking block";

#[derive(Debug)]
pub struct ThinkingKeyError(String);

impl fmt::Display for ThinkingKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ThinkingKeyError {}

/// Encrypt and authenticate hidden reasoning with a Fernet key.
#[derive(Clone)]
pub struct ThinkingCodec {
    key: [u8; KEY_BYTES],
}

impl ThinkingCodec {
    /// Whether this build has a platform crypto backend for Fernet.
    pub const fn is_supported() -> bool {
        cfg!(target_os = "macos")
    }

    /// Construct a codec from a URL-safe-base64 Fernet key.
    pub fn new(key: &[u8]) -> Result<Self, ThinkingKeyError> {
        let decoded = URL_SAFE
            .decode(key)
            .map_err(|_| ThinkingKeyError("Invalid thinking key".into()))?;
        let key: [u8; KEY_BYTES] = decoded
            .try_into()
            .map_err(|_| ThinkingKeyError("Invalid thinking key".into()))?;
        Ok(Self { key })
    }

    /// Generate an ephemeral key when the caller does not persist one.
    pub fn generate() -> Result<Self, ThinkingKeyError> {
        let key = generate_key()?;
        Self::new(&key)
    }

    /// Return the standard padded Fernet key encoding.
    pub fn key(&self) -> String {
        URL_SAFE.encode(self.key)
    }

    pub fn encode(&self, text: &str) -> Result<String, ApiError> {
        if !Self::is_supported() {
            return Err(crypto_unavailable());
        }
        let payload = text.as_bytes();
        let estimated = 4 * ((57 + 16 * (payload.len() / 16 + 1) + 2) / 3);
        if estimated > MAX_SIGNATURE_BYTES {
            return Err(ApiError::new(
                500,
                "thinking exceeds the signature size limit",
                "thinking_too_large",
            ));
        }
        let mut iv = [0u8; 16];
        random_bytes(&mut iv).map_err(|_| {
            ApiError::new(500, "cannot generate thinking signature", "thinking_error")
        })?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut token = Vec::with_capacity(1 + 8 + 16 + payload.len() + 16 + 32);
        token.push(0x80);
        token.extend_from_slice(&timestamp.to_be_bytes());
        token.extend_from_slice(&iv);
        token.extend_from_slice(&aes_cbc(true, &self.key[16..], &iv, payload).map_err(|_| {
            ApiError::new(500, "cannot encrypt thinking signature", "thinking_error")
        })?);
        let mac = hmac_sha256(&self.key[..16], &token).map_err(|_| {
            ApiError::new(500, "cannot authenticate thinking signature", "thinking_error")
        })?;
        token.extend_from_slice(&mac);
        Ok(URL_SAFE.encode(token))
    }

    pub fn decode(&self, signature: &str) -> Result<String, ApiError> {
        if !Self::is_supported() {
            return Err(crypto_unavailable());
        }
        if signature.len() > MAX_SIGNATURE_BYTES {
            return Err(invalid_signature());
        }
        let token = URL_SAFE
            .decode(signature.as_bytes())
            .map_err(|_| invalid_signature())?;
        // Header + IV + one AES block + HMAC.
        if token.len() < 1 + 8 + 16 + 16 + 32 || token[0] != 0x80 {
            return Err(invalid_signature());
        }
        let authenticated_len = token.len() - 32;
        let expected = hmac_sha256(&self.key[..16], &token[..authenticated_len])
            .map_err(|_| invalid_signature())?;
        if !constant_time_eq(&expected, &token[authenticated_len..]) {
            return Err(invalid_signature());
        }
        let iv: [u8; 16] = token[9..25]
            .try_into()
            .map_err(|_| invalid_signature())?;
        let ciphertext = &token[25..authenticated_len];
        if ciphertext.is_empty() || ciphertext.len() % 16 != 0 {
            return Err(invalid_signature());
        }
        let plaintext = aes_cbc(false, &self.key[16..], &iv, ciphertext)
            .map_err(|_| invalid_signature())?;
        String::from_utf8(plaintext).map_err(|_| invalid_signature())
    }
}

/// Load (or atomically create) the user's persistent Fernet key.
pub fn load_thinking_key(path: Option<&Path>) -> Result<Vec<u8>, ThinkingKeyError> {
    let path = match path {
        Some(path) => path.to_path_buf(),
        None => default_key_path()?,
    };
    match read_key(&path) {
        Ok(key) => return Ok(key),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(map_key_io(&path, error)),
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut directory = fs::DirBuilder::new();
    directory.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        directory.mode(0o700);
    }
    directory
        .create(parent)
        .map_err(|error| map_key_io(&path, error))?;
    let raw = random_array::<KEY_BYTES>().map_err(|error| map_key_io(&path, error))?;
    let encoded = URL_SAFE.encode(raw);
    let temp = unique_temp_path(parent);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp).map_err(|error| map_key_io(&path, error))?;
    let result = (|| {
        file.write_all(encoded.as_bytes())?;
        file.sync_all()?;
        // Hard-linking publishes the fully-written file without replacing a
        // key another server process may have created at the same time.
        match fs::hard_link(&temp, &path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => Err(error),
        }
    })();
    drop(file);
    let _ = fs::remove_file(&temp);
    result.map_err(|error| map_key_io(&path, error))?;
    read_key(&path).map_err(|error| map_key_io(&path, error))
}

fn read_key(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if !metadata.is_file()
            || metadata.uid() != unsafe { libc::getuid() }
            || metadata.mode() & 0o7777 != 0o400 && metadata.mode() & 0o7777 != 0o600
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "thinking key must be a user-owned regular file with mode 0400 or 0600",
            ));
        }
    }
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "thinking key is not a regular file",
        ));
    }
    let mut key = Vec::with_capacity(KEY_ENCODED_BYTES + 1);
    file.take((KEY_ENCODED_BYTES + 1) as u64).read_to_end(&mut key)?;
    if key.len() != KEY_ENCODED_BYTES || ThinkingCodec::new(&key).is_err() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid thinking key",
        ));
    }
    Ok(key)
}

fn default_key_path() -> Result<PathBuf, ThinkingKeyError> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| ThinkingKeyError("Cannot locate home directory".into()))?;
    Ok(PathBuf::from(home)
        .join("Library/Application Support/Splash/thinking.key"))
}

fn unique_temp_path(parent: &Path) -> PathBuf {
    let mut nonce = [0u8; 12];
    let _ = random_bytes(&mut nonce);
    let suffix = nonce.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    parent.join(format!(".thinking-{}-{suffix}", std::process::id()))
}

fn map_key_io(path: &Path, error: std::io::Error) -> ThinkingKeyError {
    ThinkingKeyError(format!("Cannot use thinking key {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "MDEyMzQ1Njc4OWFiY2RlZmZlZGNiYTk4NzY1NDMyMTA=";
    // Produced by Python cryptography.Fernet with timestamp 1700000000 and a
    // fixed IV, so changes to the token layout or MAC input break this test.
    const PYTHON_TOKEN: &str = "gAAAAABlU_EAMTIzNDU2Nzg5MGFiY2RlZieqP-OSnO4lP4dFAUiD_XEGtjO1UuujG2Mhrj5Gg5meQjK619luWldQuecsHRdKpY1DavOt9WBiwUG8xaYNNAY=";

    #[test]
    fn decodes_python_fernet_token() {
        let codec = ThinkingCodec::new(KEY.as_bytes()).expect("valid Fernet key");
        assert_eq!(codec.decode(PYTHON_TOKEN).expect("valid token"), "private reasoning 🧠");
    }

    #[test]
    fn generated_token_is_accepted_by_python_cryptography_when_available() {
        let codec = ThinkingCodec::new(KEY.as_bytes()).expect("valid Fernet key");
        let token = codec.encode("Rust reasoning 🦀").expect("encoded token");
        let script = "from cryptography.fernet import Fernet; import sys; print(Fernet(sys.argv[1].encode()).decrypt(sys.argv[2].encode()).decode())";
        let output = match std::process::Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(KEY)
            .arg(token)
            .output()
        {
            Ok(output) => output,
            Err(_) => return, // Python is optional for Rust-only builds.
        };
        if !output.status.success() {
            // Python may be installed without the project's cryptography dep.
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No module named 'cryptography'") {
                return;
            }
            panic!("Python Fernet rejected Rust token: {stderr}");
        }
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "Rust reasoning 🦀");
    }

    #[test]
    fn rejects_tampered_tokens_and_wrong_keys() {
        let codec = ThinkingCodec::new(KEY.as_bytes()).expect("valid Fernet key");
        let mut token = PYTHON_TOKEN.to_owned();
        let replacement = if &token[20..21] == "A" { "B" } else { "A" };
        token.replace_range(20..21, replacement);
        assert_eq!(codec.decode(&token).unwrap_err().code, "invalid_thinking_signature");
        let wrong_key = ThinkingCodec::new(URL_SAFE.encode([9u8; KEY_BYTES]).as_bytes()).unwrap();
        assert!(wrong_key.decode(PYTHON_TOKEN).is_err());
    }

    #[test]
    fn loads_and_persists_a_private_fernet_key() {
        let root = std::env::temp_dir().join(format!(
            "splash-thinking-key-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = root.join("nested/thinking.key");
        let first = load_thinking_key(Some(&path)).expect("create key");
        let second = load_thinking_key(Some(&path)).expect("reuse key");
        assert_eq!(first, second);
        assert_eq!(first.len(), KEY_ENCODED_BYTES);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        }
        fs::remove_dir_all(root).expect("remove temporary key directory");
    }
}

fn generate_key() -> Result<Vec<u8>, ThinkingKeyError> {
    Ok(URL_SAFE.encode(random_array::<KEY_BYTES>().map_err(|e| ThinkingKeyError(e.to_string()))?).into_bytes())
}

fn random_array<const N: usize>() -> std::io::Result<[u8; N]> {
    let mut bytes = [0u8; N];
    random_bytes(&mut bytes)?;
    Ok(bytes)
}

fn random_bytes(bytes: &mut [u8]) -> std::io::Result<()> {
    File::open("/dev/urandom")?.read_exact(bytes)
}

fn invalid_signature() -> ApiError {
    ApiError::new(
        400,
        format!("{INVALID_SIGNATURE}; hidden thinking must be returned to a server using the key that generated it"),
        "invalid_thinking_signature",
    )
}

pub fn crypto_unavailable() -> ApiError {
    ApiError::new(
        501,
        "thinking signatures are unavailable on this platform",
        "thinking_crypto_unavailable",
    )
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

#[cfg(target_os = "macos")]
fn aes_cbc(encrypt: bool, key: &[u8], iv: &[u8; 16], input: &[u8]) -> Result<Vec<u8>, ()> {
    use std::ffi::c_void;
    #[link(name = "System")]
    unsafe extern "C" {
        fn CCCrypt(
            op: u32,
            algorithm: u32,
            options: u32,
            key: *const c_void,
            key_len: usize,
            iv: *const c_void,
            data_in: *const c_void,
            data_in_len: usize,
            data_out: *mut c_void,
            data_out_available: usize,
            data_out_moved: *mut usize,
        ) -> i32;
    }
    let mut output = vec![0u8; input.len() + 16];
    let mut moved = 0usize;
    // CommonCrypto constants: AES=0, PKCS7 padding=1, encrypt=0, decrypt=1.
    let status = unsafe {
        CCCrypt(
            if encrypt { 0 } else { 1 },
            0,
            1,
            key.as_ptr().cast(),
            key.len(),
            iv.as_ptr().cast(),
            input.as_ptr().cast(),
            input.len(),
            output.as_mut_ptr().cast(),
            output.len(),
            &mut moved,
        )
    };
    if status != 0 {
        return Err(());
    }
    output.truncate(moved);
    Ok(output)
}

#[cfg(not(target_os = "macos"))]
fn aes_cbc(_encrypt: bool, _key: &[u8], _iv: &[u8; 16], _input: &[u8]) -> Result<Vec<u8>, ()> {
    Err(())
}

#[cfg(target_os = "macos")]
fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>, ()> {
    use std::ffi::c_void;
    #[link(name = "System")]
    unsafe extern "C" {
        fn CCHmac(
            algorithm: u32,
            key: *const c_void,
            key_len: usize,
            data: *const c_void,
            data_len: usize,
            mac_out: *mut c_void,
        );
    }
    let mut digest = vec![0u8; 32];
    // CommonCrypto kCCHmacAlgSHA256 = 2.
    unsafe {
        CCHmac(
            2,
            key.as_ptr().cast(),
            key.len(),
            data.as_ptr().cast(),
            data.len(),
            digest.as_mut_ptr().cast(),
        )
    };
    Ok(digest)
}

#[cfg(not(target_os = "macos"))]
fn hmac_sha256(_key: &[u8], _data: &[u8]) -> Result<Vec<u8>, ()> {
    Err(())
}
