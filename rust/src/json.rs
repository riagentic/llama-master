//! Minimal JSON writer — the only serialization this crate needs.
//!
//! Deliberately dependency-free (no serde): every parser here emits a flat,
//! known-shape object, so a 40-line escaper beats a 300 KB derive macro in a
//! WASM artifact that ships inside the app bundle.

/// Escape a string into a JSON string literal, including the surrounding quotes.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Format an f64 so the result is always valid JSON (NaN/Inf are not).
pub fn num(v: f64) -> String {
    if v.is_finite() {
        // Trim a trailing ".0" so integers stay integers in the JSON.
        let s = format!("{}", v);
        s
    } else {
        "0".to_string()
    }
}

/// `{"error":"..."}` — the single failure shape every export returns.
pub fn error(msg: &str) -> String {
    format!("{{\"ok\":false,\"error\":{}}}", quote(msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_control_and_quotes() {
        assert_eq!(quote("a\"b"), "\"a\\\"b\"");
        assert_eq!(quote("a\nb"), "\"a\\nb\"");
        assert_eq!(quote("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn non_finite_numbers_never_break_json() {
        assert_eq!(num(f64::NAN), "0");
        assert_eq!(num(f64::INFINITY), "0");
        assert_eq!(num(1.5), "1.5");
    }
}
