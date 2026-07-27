//! llama-sys — the pure-computation core of llama.master, compiled to WASM.
//!
//! Everything here is a total function from bytes to JSON. No `std::fs`, no
//! `std::process`, no clock, no network: the Deno host performs every read and
//! spawn and hands the raw text in. That split is what makes the hard parts
//! (GGUF byte accounting, telemetry parsing) unit-testable with `cargo test`
//! and identical in the app, in tests, and on any platform.
//!
//! ABI (mirrors the host loader in `src/cell/wasm.server.ts`):
//!   host: `alloc(len)` → write bytes → `fn(ptr, len, …)` → returns a pointer to
//!   a 4-byte little-endian length followed by UTF-8 JSON → host reads it via
//!   `str_len(ptr)`, copies it out, then `free_str(ptr)` and `dealloc(ptr, len)`
//!   for every input buffer.

pub mod gguf;
pub mod json;
pub mod sys;

use std::alloc::{alloc as rust_alloc, dealloc as rust_dealloc, Layout};

// ── memory protocol ────────────────────────────────────────────────────────

/// Reserve `len` bytes for the host to write an input buffer into.
///
/// # Safety
/// The host must pass the same `len` back to [`dealloc`].
#[no_mangle]
pub unsafe extern "C" fn alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align_unchecked(len, 1);
    rust_alloc(layout)
}

/// Release a buffer obtained from [`alloc`].
///
/// # Safety
/// `ptr`/`len` must be exactly what [`alloc`] returned and was called with.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    rust_dealloc(ptr, Layout::from_size_align_unchecked(len, 1));
}

/// Length of a returned length-prefixed string, in bytes (excluding the prefix).
///
/// # Safety
/// `ptr` must come from one of this module's JSON-returning exports.
#[no_mangle]
pub unsafe extern "C" fn str_len(ptr: *const u8) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    u32::from_le_bytes([*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)])
}

/// Release a returned length-prefixed string.
///
/// # Safety
/// `ptr` must come from one of this module's JSON-returning exports and must
/// not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn free_str(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let len = str_len(ptr) as usize;
    let total = len + 4;
    rust_dealloc(ptr, Layout::from_size_align_unchecked(total, 1));
}

/// Move a Rust `String` into host-readable memory as `[len: u32 LE][utf8]`.
fn out(s: String) -> *mut u8 {
    let bytes = s.into_bytes();
    let total = bytes.len() + 4;
    unsafe {
        let layout = Layout::from_size_align_unchecked(total, 1);
        let p = rust_alloc(layout);
        if p.is_null() {
            return p;
        }
        std::ptr::copy_nonoverlapping((bytes.len() as u32).to_le_bytes().as_ptr(), p, 4);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(4), bytes.len());
        p
    }
}

/// # Safety
/// `ptr` must point to `len` readable bytes (or be null with `len == 0`).
unsafe fn input<'a>(ptr: *const u8, len: usize) -> &'a str {
    if ptr.is_null() || len == 0 {
        return "";
    }
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len)).unwrap_or("")
}

// ── exports ────────────────────────────────────────────────────────────────

/// Parse a GGUF header prefix. Returns the model JSON, or
/// `{"ok":false,"error":"truncated","need":N}` when the prefix was too short —
/// the host then re-reads `N` bytes and calls again (see `models.server.ts`).
///
/// # Safety
/// `ptr`/`len` must describe a readable byte range.
#[no_mangle]
pub unsafe extern "C" fn gguf_parse(ptr: *const u8, len: usize) -> *mut u8 {
    if ptr.is_null() || len == 0 {
        return out(json::error("empty"));
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    out(match gguf::parse(bytes) {
        Ok(g) => gguf::to_json(&g),
        Err(need) if need != usize::MAX => {
            format!("{{\"ok\":false,\"error\":\"truncated\",\"need\":{}}}", need)
        }
        Err(_) => json::error("not a readable GGUF header"),
    })
}

/// # Safety
/// Each `ptr`/`len` pair must describe a readable byte range.
#[no_mangle]
pub unsafe extern "C" fn sys_cpu(
    cpuinfo: *const u8,
    cpuinfo_len: usize,
    stat: *const u8,
    stat_len: usize,
    hwmon: *const u8,
    hwmon_len: usize,
) -> *mut u8 {
    out(sys::cpu(
        input(cpuinfo, cpuinfo_len),
        input(stat, stat_len),
        input(hwmon, hwmon_len),
    ))
}

/// # Safety
/// `ptr`/`len` must describe a readable byte range.
#[no_mangle]
pub unsafe extern "C" fn sys_mem(ptr: *const u8, len: usize) -> *mut u8 {
    out(sys::mem(input(ptr, len)))
}

/// # Safety
/// Each `ptr`/`len` pair must describe a readable byte range.
#[no_mangle]
pub unsafe extern "C" fn sys_gpu(
    nvidia: *const u8,
    nvidia_len: usize,
    sysfs: *const u8,
    sysfs_len: usize,
) -> *mut u8 {
    out(sys::gpu(input(nvidia, nvidia_len), input(sysfs, sysfs_len)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_prefixes_the_length_and_free_str_reads_it_back() {
        let p = out("hello".to_string());
        unsafe {
            assert_eq!(str_len(p), 5);
            let body = std::slice::from_raw_parts(p.add(4), 5);
            assert_eq!(body, b"hello");
            free_str(p);
        }
    }

    #[test]
    fn gguf_parse_reports_truncation_instead_of_failing() {
        let mut header = b"GGUF".to_vec();
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        let p = unsafe { gguf_parse(header.as_ptr(), header.len()) };
        let s = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(p.add(4), str_len(p) as usize))
                .unwrap()
                .to_string()
        };
        unsafe { free_str(p) };
        assert!(s.contains("\"truncated\""), "{}", s);
    }

    #[test]
    fn empty_input_is_an_error_not_a_panic() {
        let p = unsafe { gguf_parse(std::ptr::null(), 0) };
        let s = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(p.add(4), str_len(p) as usize))
                .unwrap()
                .to_string()
        };
        unsafe { free_str(p) };
        assert!(s.contains("\"ok\":false"), "{}", s);
    }
}
