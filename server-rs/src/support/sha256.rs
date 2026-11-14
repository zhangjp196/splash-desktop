//! SHA-256, as the installer names a selection link: the digest of the
//! selection's compact JSON. The server resolves a prepared model the way
//! `install/models.py` records it, so it needs the same digest without a
//! dependency the runtime does not otherwise carry.

use std::io::{self, Read};

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The SHA-256 digest of `bytes`, as raw bytes.
pub fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut state = H0;
    let mut message = bytes.to_vec();
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());
    for block in message.chunks_exact(64) {
        compress(&mut state, block);
    }
    let mut digest = [0u8; 32];
    for (slot, word) in digest.chunks_exact_mut(4).zip(state) {
        slot.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// The SHA-256 digest of `bytes`, as lowercase hexadecimal.
pub fn hex_digest(bytes: &[u8]) -> String {
    digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Stream a file through SHA-256 without allocating memory proportional to
/// model size.
pub fn hash_reader(mut input: impl Read) -> io::Result<String> {
    let mut state = H0;
    let mut pending = Vec::with_capacity(8192);
    let mut total = 0u64;
    let mut chunk = [0u8; 8192];
    loop {
        let count = input.read(&mut chunk)?;
        if count == 0 { break; }
        total = total.checked_add(count as u64).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "input is too large to hash"))?;
        pending.extend_from_slice(&chunk[..count]);
        let complete = pending.len() / 64 * 64;
        for block in pending[..complete].chunks_exact(64) { compress(&mut state, block); }
        pending.drain(..complete);
    }
    pending.push(0x80);
    while pending.len() % 64 != 56 { pending.push(0); }
    pending.extend_from_slice(&total.wrapping_mul(8).to_be_bytes());
    for block in pending.chunks_exact(64) { compress(&mut state, block); }
    let mut digest = [0u8; 32];
    for (slot, word) in digest.chunks_exact_mut(4).zip(state) { slot.copy_from_slice(&word.to_be_bytes()); }
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn compress(state: &mut [u32; 8], block: &[u8]) {
    let mut schedule = [0u32; 64];
    for (index, word) in block.chunks_exact(4).enumerate() {
        schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choice = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(choice)
            .wrapping_add(K[index])
            .wrapping_add(schedule[index]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}

#[cfg(test)]
mod tests {
    use super::{hash_reader, hex_digest};
    use std::io::Cursor;

    #[test]
    fn the_digest_matches_the_published_vectors() {
        assert_eq!(
            hex_digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn streaming_hash_matches_buffered_hash_across_block_boundaries() {
        for size in [0, 1, 55, 56, 63, 64, 65, 8191, 8192, 8193, 100_003] {
            let bytes: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            assert_eq!(hash_reader(Cursor::new(&bytes)).unwrap(), hex_digest(&bytes));
        }
    }

    #[test]
    fn a_selection_link_matches_the_installers_digest() {
        // install/models.py's local_selection_link for a directory with no
        // options: json.dumps([...], separators=(",", ":")).
        let selection = r#"["/Users/zhangjp/Documents/ai-models/mlx/CaseD0rsett/Qwen3.8-4B-Distill-Heretic-Abliterated-MLX-4bit",false,null,"auto",false]"#;
        assert_eq!(
            hex_digest(selection.as_bytes()),
            "16691435ba0c642719ccb4995ab2a72f2e90fa64b0cd34d9fbab001932d3c6f5"
        );
    }
}
