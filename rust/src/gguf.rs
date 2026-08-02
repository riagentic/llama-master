//! GGUF header parser — pure bytes in, JSON out.
//!
//! Why this exists: every memory number the UI shows (what fits in VRAM, what
//! spills to RAM, what a `--n-cpu-moe` split actually saves) is derived from the
//! EXACT byte size of every tensor, grouped per transformer layer. Estimating
//! from the file size divided by layer count is wrong for MoE models by a factor
//! of five — the routed experts dominate, and they are exactly the tensors the
//! planner wants to move to CPU.
//!
//! Format (ggml-org/llama.cpp, `docs/gguf.md`):
//!   magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
//!   kv*   : key string | value_type u32 | value
//!   info* : name string | n_dims u32 | dims[u64] | ggml_type u32 | offset u64
//! Strings are `u64 len + utf8` (v1 used u32 — handled).
//!
//! The parser never allocates the tensor list: it folds each entry into a
//! per-layer accumulator as it walks, so a 700-tensor 70B header costs ~4 KB.

use crate::json::{num, quote};

// ── ggml type table ────────────────────────────────────────────────────────
// (block_size, bytes_per_block). Ids are ggml_type from ggml.h; the gaps are
// types that were removed upstream (4,5 = Q4_2/Q4_3; 31..33, 36..38 = the
// repacked Q4_0_M_N / IQ4_NL_M_N families). An id we do not know is reported
// rather than guessed — a wrong size here is a wrong VRAM bar.
fn type_info(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        _ => return None,
    })
}

fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => "UNKNOWN",
    }
}

/// `general.file_type` (LLAMA_FTYPE) → the quant label users recognise.
fn ftype_name(f: u32) -> Option<&'static str> {
    Some(match f {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}

// ── cursor ─────────────────────────────────────────────────────────────────

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
    /// GGUF v1 encoded string/array lengths as u32; v2+ uses u64.
    short_len: bool,
}

/// `Err(need)` = the header is longer than the slice we were given; `need` is a
/// lower bound on the bytes required. The host re-reads and retries — truncation
/// is a normal first-attempt outcome, never an error the user sees.
type R<T> = Result<T, usize>;

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or(usize::MAX)?;
        if end > self.b.len() {
            return Err(end);
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> R<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> R<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }
    fn len(&mut self) -> R<u64> {
        if self.short_len {
            Ok(self.u32()? as u64)
        } else {
            self.u64()
        }
    }
    fn str(&mut self) -> R<String> {
        let n = self.len()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
    fn skip(&mut self, n: usize) -> R<()> {
        self.take(n).map(|_| ())
    }
}

// ── metadata values ────────────────────────────────────────────────────────

/// The subset of a KV value the planner cares about: a number or a string.
/// Arrays are skipped (the tokenizer vocab is 99% of a GGUF header by volume).
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Num(f64),
    Str(String),
    Skipped,
}

fn fixed_size(t: u32) -> Option<usize> {
    Some(match t {
        0 | 1 | 7 => 1, // u8 / i8 / bool
        2 | 3 => 2,     // u16 / i16
        4 | 5 | 6 => 4, // u32 / i32 / f32
        10 | 11 | 12 => 8,
        _ => return None,
    })
}

fn read_value(c: &mut Cur, t: u32) -> R<Val> {
    Ok(match t {
        0 => Val::Num(c.take(1)?[0] as f64),
        1 => Val::Num(c.take(1)?[0] as i8 as f64),
        2 => {
            let s = c.take(2)?;
            Val::Num(u16::from_le_bytes([s[0], s[1]]) as f64)
        }
        3 => {
            let s = c.take(2)?;
            Val::Num(i16::from_le_bytes([s[0], s[1]]) as f64)
        }
        4 => Val::Num(c.u32()? as f64),
        5 => Val::Num(c.u32()? as i32 as f64),
        6 => Val::Num(f32::from_bits(c.u32()?) as f64),
        7 => Val::Num(c.take(1)?[0] as f64),
        8 => Val::Str(c.str()?),
        9 => {
            let it = c.u32()?;
            let n = c.len()? as usize;
            if let Some(sz) = fixed_size(it) {
                c.skip(n.saturating_mul(sz))?;
            } else if it == 8 {
                for _ in 0..n {
                    let l = c.len()? as usize;
                    c.skip(l)?;
                }
            } else {
                // Nested arrays are not produced by any converter in the wild;
                // bail loudly rather than silently mis-seeking the rest.
                return Err(usize::MAX);
            }
            Val::Skipped
        }
        10 => Val::Num(c.u64()? as f64),
        11 => Val::Num(c.u64()? as i64 as f64),
        12 => {
            let s = c.take(8)?;
            Val::Num(f64::from_le_bytes([
                s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            ]))
        }
        _ => return Err(usize::MAX),
    })
}

// ── the parse ──────────────────────────────────────────────────────────────

/// Byte footprint of one transformer block, split so the planner can move the
/// routed experts to CPU independently of attention.
#[derive(Default, Clone, Copy)]
pub struct Layer {
    pub bytes: u64,
    pub expert_bytes: u64,
}

/// Everything the planner and the UI need from a GGUF file.
pub struct Gguf {
    pub version: u32,
    pub arch: String,
    pub name: String,
    pub quant: String,
    pub n_layer: usize,
    pub n_ctx_train: u64,
    pub n_embd: u64,
    pub n_head: u64,
    pub n_head_kv: u64,
    pub key_length: u64,
    pub value_length: u64,
    /// Interleaved sliding-window attention: the window, in tokens, and how
    /// often a full-attention layer appears. Gemma-3 caches 1024 tokens on five
    /// layers out of every six, so billing every layer for the whole context
    /// overstates its KV cache several-fold. 0 = every layer is full attention.
    pub swa_window: u64,
    pub swa_pattern: u64,
    /// Multi-head latent attention (DeepSeek-V2/V3): the cache holds a
    /// compressed latent of this rank plus the RoPE part, not one entry per
    /// head, which is a ~70x difference on V3. 0 = not an MLA model.
    pub kv_lora_rank: u64,
    /// Multi-token-prediction blocks (`<arch>.nextn_predict_layers`). A model
    /// that declares these ships an extra block that can DRAFT the next tokens,
    /// which llama.cpp verifies against the full model — speculative decoding
    /// that needs no second model and changes no output. `block_count` INCLUDES
    /// them, and llama.cpp's own `n_layer()` subtracts them again, so the count
    /// is only meaningful alongside this. 0 = not an MTP model.
    pub nextn_layers: u64,
    pub n_expert: u64,
    pub n_expert_used: u64,
    pub rope_freq_base: f64,
    pub n_tensors: u64,
    pub tensor_bytes: u64,
    /// `token_embd.*` — the input lookup table. Split out from `output_bytes`
    /// because the two land on different devices at partial offload.
    pub embd_bytes: u64,
    /// `output.*` / `output_norm.*` — offloaded only when `-ngl` exceeds the
    /// layer count.
    pub output_bytes: u64,
    pub unknown_types: u64,
    pub layers: Vec<Layer>,
    /// This part's index, and how many parts the model has (`split.no`,
    /// `split.count`). A model over ~40 GB is always shipped split, and each
    /// part carries ONLY its own slice of the tensor table — so everything
    /// above (`tensor_bytes`, `layers`, `embd_bytes`, `output_bytes`) describes
    /// this part alone until the parts are merged. 0 = not a split model.
    /// Sparse-attention indexer (`<arch>.attention.indexer.top_k`, DeepSeek-V4's
    /// "lightning indexer"). Non-zero means the graph scores the WHOLE context
    /// for every micro-batch token, so the compute buffer grows with the context
    /// instead of being a flat per-process cost — 68.5 GiB measured at a
    /// 1,048,576 context where the flat estimate said 730 MB. 0 = ordinary
    /// attention, where the flat estimate holds.
    /// `<arch>.rope.scaling.original_context_length` — the length the model was
    /// ACTUALLY trained at, before RoPE scaling stretched the advertised figure.
    /// DeepSeek-V4-Flash declares 1,048,576 and this says 65,536: the headline is
    /// a 16x YaRN extrapolation. 0 = no scaling, the headline is the truth.
    pub n_ctx_orig: u64,
    pub indexer_top_k: u64,
    pub split_no: u64,
    pub split_count: u64,
    /// Tensors across ALL parts (`split.tensors.count`). The check that a merge
    /// actually saw everything: reading part 1 of four gives 38 of 1328.
    pub split_tensors: u64,
}

fn kv_num(kv: &[(String, Val)], key: &str) -> Option<f64> {
    kv.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Val::Num(n) => Some(*n),
        _ => None,
    })
}

fn kv_str(kv: &[(String, Val)], key: &str) -> Option<String> {
    kv.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Val::Str(s) => Some(s.clone()),
        _ => None,
    })
}

/// Layer index out of `blk.<N>.…` (the universal llama.cpp naming).
fn layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (n, _) = rest.split_once('.')?;
    n.parse().ok()
}

/// Ceiling on the layer table, so a hostile `blk.4000000000.weight` cannot ask
/// for an allocation instead of being ignored. No model is within three orders
/// of magnitude of it.
const MAX_LAYERS: usize = 100_000;

/// Routed-expert tensors — the ones `--n-cpu-moe` / `-ot` move off the GPU.
/// `*_shexp` (shared expert) runs for every token and stays with attention.
fn is_expert(name: &str) -> bool {
    name.contains("_exps")
}

pub fn parse(bytes: &[u8]) -> Result<Gguf, usize> {
    let mut c = Cur {
        b: bytes,
        p: 0,
        short_len: false,
    };
    if c.take(4)? != b"GGUF" {
        return Err(usize::MAX);
    }
    let version = c.u32()?;
    if version == 0 || version > 3 {
        return Err(usize::MAX);
    }
    c.short_len = version == 1;
    let n_tensors = c.len()?;
    let n_kv = c.len()?;
    if n_tensors > 1_000_000 || n_kv > 100_000 {
        return Err(usize::MAX);
    }

    let mut kv: Vec<(String, Val)> = Vec::new();
    for _ in 0..n_kv {
        let key = c.str()?;
        let t = c.u32()?;
        let v = read_value(&mut c, t)?;
        // Keep only scalars/strings — arrays are already skipped by value.
        if !matches!(v, Val::Skipped) {
            kv.push((key, v));
        }
    }

    let arch = kv_str(&kv, "general.architecture").unwrap_or_else(|| "unknown".into());
    let a = |suffix: &str| -> Option<f64> { kv_num(&kv, &format!("{}.{}", arch, suffix)) };

    let n_layer = a("block_count").unwrap_or(0.0) as usize;
    let n_embd = a("embedding_length").unwrap_or(0.0) as u64;
    let n_head = a("attention.head_count").unwrap_or(0.0) as u64;
    let n_head_kv = a("attention.head_count_kv").unwrap_or(n_head as f64) as u64;
    let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };
    let key_length = a("attention.key_length").unwrap_or(head_dim as f64) as u64;
    let value_length = a("attention.value_length").unwrap_or(head_dim as f64) as u64;
    // Absent on the great majority of models, and absent means "full attention
    // on every layer" — the formula the planner already had.
    let swa_window = a("attention.sliding_window").unwrap_or(0.0) as u64;
    // llama.cpp's default when a window is declared without a pattern is 1,
    // i.e. every layer is local.
    let swa_pattern = a("attention.sliding_window_pattern").unwrap_or(1.0) as u64;
    let kv_lora_rank = a("attention.kv_lora_rank").unwrap_or(0.0) as u64;

    let mut layers = vec![Layer::default(); n_layer];
    let mut embd_bytes: u64 = 0;
    let mut output_bytes: u64 = 0;
    let mut tensor_bytes: u64 = 0;
    let mut unknown_types: u64 = 0;
    let mut type_hist: Vec<(u32, u64)> = Vec::new();

    for _ in 0..n_tensors {
        let name = c.str()?;
        let n_dims = c.u32()?;
        if n_dims > 8 {
            return Err(usize::MAX);
        }
        let mut elems: u64 = 1;
        for _ in 0..n_dims {
            elems = elems.saturating_mul(c.u64()?);
        }
        let t = c.u32()?;
        let _offset = c.u64()?;

        let size = match type_info(t) {
            Some((block, per_block)) if block > 0 => elems / block * per_block,
            _ => {
                unknown_types += 1;
                0
            }
        };
        tensor_bytes = tensor_bytes.saturating_add(size);
        match type_hist.iter_mut().find(|(ty, _)| *ty == t) {
            Some((_, n)) => *n += size,
            None => type_hist.push((t, size)),
        }

        match layer_index(&name) {
            // A `blk.N` past `block_count` is not a lie to be filed under
            // "output". Parts 2..N of a split model carry THREE metadata keys —
            // `split.no`, `split.count`, `split.tensors.count` — and no
            // `block_count` at all, so every layer they hold is out of range of
            // a table sized from the header. Billing those to `output_bytes`
            // put 107 GB of DeepSeek-V4's experts where `-ngl` cannot move them
            // and `--n-cpu-moe` cannot see them. The table grows to whatever the
            // tensors actually say; `n_layer` stays the model's own count, so
            // the KV geometry is untouched.
            Some(i) if i < MAX_LAYERS => {
                if i >= layers.len() {
                    layers.resize(i + 1, Layer::default());
                }
                layers[i].bytes += size;
                if is_expert(&name) {
                    layers[i].expert_bytes += size;
                }
            }
            _ if name.starts_with("token_embd") => embd_bytes += size,
            _ => output_bytes += size,
        }
    }

    // Quant label: the declared file_type when we know it, else the type that
    // owns the most bytes (which is what a human would call the quant anyway).
    let quant = kv_num(&kv, "general.file_type")
        .and_then(|f| ftype_name(f as u32))
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            type_hist
                .iter()
                .max_by_key(|(_, n)| *n)
                .map(|(t, _)| type_name(*t).to_string())
                .unwrap_or_else(|| "UNKNOWN".into())
        });

    let indexer_top_k = a("attention.indexer.top_k").unwrap_or(0.0) as u64;
    let n_ctx_orig = a("rope.scaling.original_context_length").unwrap_or(0.0) as u64;

    Ok(Gguf {
        version,
        name: kv_str(&kv, "general.name").unwrap_or_default(),
        quant,
        n_layer,
        n_ctx_train: a("context_length").unwrap_or(0.0) as u64,
        n_embd,
        n_head,
        n_head_kv,
        key_length,
        value_length,
        swa_window,
        swa_pattern,
        kv_lora_rank,
        nextn_layers: a("nextn_predict_layers").unwrap_or(0.0) as u64,
        n_expert: a("expert_count").unwrap_or(0.0) as u64,
        n_expert_used: a("expert_used_count").unwrap_or(0.0) as u64,
        rope_freq_base: a("rope.freq_base").unwrap_or(0.0),
        n_tensors,
        tensor_bytes,
        embd_bytes,
        output_bytes,
        unknown_types,
        layers,
        arch,
        // NOT arch-prefixed: `split.*` describes the file, not the model.
        n_ctx_orig,
        indexer_top_k,
        split_no: kv_num(&kv, "split.no").unwrap_or(0.0) as u64,
        split_count: kv_num(&kv, "split.count").unwrap_or(0.0) as u64,
        split_tensors: kv_num(&kv, "split.tensors.count").unwrap_or(0.0) as u64,
    })
}

pub fn to_json(g: &Gguf) -> String {
    let layers = g
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| format!("{{\"i\":{},\"bytes\":{},\"expert\":{}}}", i, l.bytes, l.expert_bytes))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{\"ok\":true,\"version\":{},\"arch\":{},\"name\":{},\"quant\":{},",
            "\"nLayer\":{},\"nCtxTrain\":{},\"nEmbd\":{},\"nHead\":{},\"nHeadKv\":{},",
            "\"keyLength\":{},\"valueLength\":{},",
            "\"swaWindow\":{},\"swaPattern\":{},\"kvLoraRank\":{},\"nextnLayers\":{},",
            "\"nExpert\":{},\"nExpertUsed\":{},",
            "\"ropeFreqBase\":{},\"nTensors\":{},\"tensorBytes\":{},\"embdBytes\":{},\"outputBytes\":{},",
            "\"unknownTypes\":{},\"nCtxOrig\":{},\"indexerTopK\":{},\"splitNo\":{},\"splitCount\":{},\"splitTensors\":{},",
            "\"layers\":[{}]}}"
        ),
        g.version,
        quote(&g.arch),
        quote(&g.name),
        quote(&g.quant),
        g.n_layer,
        g.n_ctx_train,
        g.n_embd,
        g.n_head,
        g.n_head_kv,
        g.key_length,
        g.value_length,
        g.swa_window,
        g.swa_pattern,
        g.kv_lora_rank,
        g.nextn_layers,
        g.n_expert,
        g.n_expert_used,
        num(g.rope_freq_base),
        g.n_tensors,
        g.tensor_bytes,
        g.embd_bytes,
        g.output_bytes,
        g.unknown_types,
        g.n_ctx_orig,
        g.indexer_top_k,
        g.split_no,
        g.split_count,
        g.split_tensors,
        layers
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a synthetic GGUF v3 header — the fixture is the format spec, so a
    // regression in the cursor arithmetic fails here rather than on a 40 GB file.
    struct Buf(Vec<u8>);
    impl Buf {
        fn new() -> Self {
            let mut b = Buf(Vec::new());
            b.0.extend_from_slice(b"GGUF");
            b.u32(3);
            b
        }
        fn u32(&mut self, v: u32) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn u64(&mut self, v: u64) {
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn s(&mut self, v: &str) {
            self.u64(v.len() as u64);
            self.0.extend_from_slice(v.as_bytes());
        }
        fn kv_str(&mut self, k: &str, v: &str) {
            self.s(k);
            self.u32(8);
            self.s(v);
        }
        fn kv_u32(&mut self, k: &str, v: u32) {
            self.s(k);
            self.u32(4);
            self.u32(v);
        }
        fn kv_str_array(&mut self, k: &str, items: &[&str]) {
            self.s(k);
            self.u32(9);
            self.u32(8);
            self.u64(items.len() as u64);
            for i in items {
                self.s(i);
            }
        }
        fn tensor(&mut self, name: &str, dims: &[u64], ty: u32) {
            self.s(name);
            self.u32(dims.len() as u32);
            for d in dims {
                self.u64(*d);
            }
            self.u32(ty);
            self.u64(0);
        }
    }

    fn fixture() -> Vec<u8> {
        let mut b = Buf::new();
        b.u64(5); // tensors
        b.u64(6); // kv
        b.kv_str("general.architecture", "llama");
        b.kv_str("general.name", "Test 7B");
        b.kv_u32("general.file_type", 15);
        b.kv_u32("llama.block_count", 2);
        b.kv_u32("llama.embedding_length", 4096);
        b.kv_str_array("tokenizer.ggml.tokens", &["a", "bb", "ccc"]);
        // 4096*4096 Q4_K = 16777216/256*144 = 9437184 bytes
        b.tensor("blk.0.attn_q.weight", &[4096, 4096], 12);
        b.tensor("blk.0.ffn_gate_exps.weight", &[4096, 4096], 12);
        b.tensor("blk.1.attn_q.weight", &[4096, 4096], 12);
        b.tensor("token_embd.weight", &[4096, 32000], 12);
        b.tensor("output_norm.weight", &[4096], 0); // F32 → 16384 bytes
        b.0
    }

    #[test]
    fn parses_metadata_and_skips_arrays() {
        let g = parse(&fixture()).expect("parses");
        assert_eq!(g.arch, "llama");
        assert_eq!(g.name, "Test 7B");
        assert_eq!(g.quant, "Q4_K_M");
        assert_eq!(g.n_layer, 2);
        assert_eq!(g.n_embd, 4096);
        assert_eq!(g.n_tensors, 5);
    }

    #[test]
    fn accounts_bytes_per_layer_and_separates_experts() {
        let g = parse(&fixture()).unwrap();
        let q4k = 4096u64 * 4096 / 256 * 144;
        assert_eq!(g.layers[0].bytes, q4k * 2);
        assert_eq!(g.layers[0].expert_bytes, q4k, "routed experts split out");
        assert_eq!(g.layers[1].bytes, q4k);
        assert_eq!(g.layers[1].expert_bytes, 0);
        let embd = 4096u64 * 32000 / 256 * 144;
        assert_eq!(g.embd_bytes, embd, "token_embd is tracked separately");
        assert_eq!(g.output_bytes, 4096 * 4, "output_norm is the output group");
        assert_eq!(g.tensor_bytes, q4k * 3 + embd + 4096 * 4);
        assert_eq!(g.unknown_types, 0);
    }

    /// A part of a split model reports how much of the model it is NOT.
    ///
    /// Without this the caller has no way to tell a whole model from 2.9% of
    /// one: part 1 of DeepSeek-V4-Flash parses cleanly and yields 38 tensors
    /// and 37 GB, of the 1328 tensors and 145 GB that are actually there.
    #[test]
    fn reports_that_it_is_one_part_of_a_split_model() {
        let mut b = Buf::new();
        b.u64(1); // tensors in THIS part
        b.u64(5); // kv
        b.kv_str("general.architecture", "llama");
        b.kv_u32("llama.block_count", 2);
        b.kv_u32("split.no", 0);
        b.kv_u32("split.count", 4);
        b.kv_u32("split.tensors.count", 1328);
        b.tensor("blk.0.attn_q.weight", &[4096, 4096], 12);
        let g = parse(&b.0).unwrap();
        assert_eq!(g.split_no, 0);
        assert_eq!(g.split_count, 4);
        assert_eq!(g.split_tensors, 1328, "the whole model's tensor count");
        assert_eq!(g.n_tensors, 1, "this part's, and only this part's");
    }

    /// Parts 2..N carry no `block_count`, and their layers must still be layers.
    ///
    /// This is the second half of the split bug: with the table sized from a
    /// header that says nothing, every `blk.N` in parts 2, 3 and 4 fell through
    /// to `output_bytes` — 107 GB of DeepSeek-V4's routed experts filed as the
    /// output head, where neither `-ngl` nor `--n-cpu-moe` can place them.
    #[test]
    fn layers_are_found_in_a_part_that_declares_no_block_count() {
        let mut b = Buf::new();
        b.u64(2);
        b.u64(3);
        b.kv_u32("split.no", 1);
        b.kv_u32("split.count", 4);
        b.kv_u32("split.tensors.count", 1328);
        b.tensor("blk.12.ffn_up_exps.weight", &[4096, 4096], 12);
        b.tensor("blk.13.attn_q.weight", &[4096, 4096], 12);
        let g = parse(&b.0).unwrap();
        let q4k = 4096u64 * 4096 / 256 * 144;
        assert_eq!(g.n_layer, 0, "the part genuinely declares none");
        assert_eq!(g.layers.len(), 14, "the table grew to what the tensors say");
        assert_eq!(g.layers[12].bytes, q4k);
        assert_eq!(g.layers[12].expert_bytes, q4k);
        assert_eq!(g.layers[13].bytes, q4k);
        assert_eq!(g.layers[13].expert_bytes, 0);
        assert_eq!(g.output_bytes, 0, "not one byte of this is the output head");
    }

    /// The overwhelming majority: one file, no `split.*` keys, no merge.
    #[test]
    fn a_single_file_model_declares_no_split() {
        let g = parse(&fixture()).unwrap();
        assert_eq!((g.split_no, g.split_count, g.split_tensors), (0, 0, 0));
    }

    #[test]
    fn truncation_reports_the_byte_count_needed() {
        let full = fixture();
        match parse(&full[..40]) {
            Err(need) => assert!(need > 40 && need != usize::MAX, "need = {}", need),
            Ok(_) => panic!("truncated header must not parse"),
        }
    }

    #[test]
    fn rejects_a_non_gguf_file() {
        assert!(parse(b"NOTAGGUF____________").is_err());
    }

    #[test]
    fn json_is_well_formed_for_odd_names() {
        let mut b = Buf::new();
        b.u64(0);
        b.u64(2);
        b.kv_str("general.architecture", "llama");
        b.kv_str("general.name", "we\"ird\nname");
        let g = parse(&b.0).unwrap();
        let j = to_json(&g);
        assert!(j.contains("we\\\"ird\\nname"));
        assert!(j.starts_with("{\"ok\":true"));
    }
}
