//! Host-telemetry parsers — raw `/proc`, `/sys` and `nvidia-smi` text in,
//! normalized JSON out. Pure: the host does every read and spawn, this crate
//! only turns bytes into numbers, which is the part worth unit-testing.

use crate::json::{num, quote};

// ── CPU ────────────────────────────────────────────────────────────────────

/// `/proc/cpuinfo` + `/proc/stat` + a host-assembled hwmon dump
/// (`label\tmillidegrees` per line) → CPU JSON.
///
/// `stat`/`coreStats` are passed through verbatim: utilization is a delta
/// between two samples, and the cell owns the previous one.
pub fn cpu(cpuinfo: &str, stat: &str, hwmon: &str) -> String {
    let mut model = String::new();
    let mut threads = 0usize;
    let mut mhz_sum = 0f64;
    let mut mhz_n = 0usize;
    // (physical id, core id) pairs — the only honest way to count real cores.
    let mut phys: Vec<(String, String)> = Vec::new();
    let mut cur_phys = String::new();

    for line in cpuinfo.lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        match k {
            "model name" if model.is_empty() => model = v.to_string(),
            "processor" => threads += 1,
            "cpu MHz" => {
                if let Ok(f) = v.parse::<f64>() {
                    mhz_sum += f;
                    mhz_n += 1;
                }
            }
            "physical id" => cur_phys = v.to_string(),
            "core id" => {
                let pair = (cur_phys.clone(), v.to_string());
                if !phys.contains(&pair) {
                    phys.push(pair);
                }
            }
            _ => {}
        }
    }

    let mut agg = String::new();
    let mut cores: Vec<String> = Vec::new();
    for line in stat.lines() {
        if line.starts_with("cpu ") {
            agg = line.to_string();
        } else if line.starts_with("cpu") && line.as_bytes().get(3).is_some_and(u8::is_ascii_digit)
        {
            cores.push(line.to_string());
        }
    }

    // Hottest package sensor wins — that is the number a user tunes against.
    let temp = hottest(hwmon);

    let core_json = cores
        .iter()
        .map(|c| quote(c))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{\"model\":{},\"cores\":{},\"threads\":{},\"mhz\":{},\"tempC\":{},",
            "\"stat\":{},\"coreStats\":[{}]}}"
        ),
        quote(&model),
        if phys.is_empty() { threads } else { phys.len() },
        threads,
        num(if mhz_n > 0 { mhz_sum / mhz_n as f64 } else { 0.0 }),
        num(temp),
        quote(&agg),
        core_json
    )
}

/// `label\tmillidegrees` lines → hottest reading in °C (0.0 = no sensor).
fn hottest(hwmon: &str) -> f64 {
    hwmon
        .lines()
        .filter_map(|l| l.rsplit_once('\t'))
        .filter_map(|(_, v)| v.trim().parse::<f64>().ok())
        .map(|m| m / 1000.0)
        // A sensor reading over 150 °C is a unit mismatch, not a hot CPU.
        .filter(|c| *c > 0.0 && *c < 150.0)
        .fold(0.0, f64::max)
}

// ── memory ─────────────────────────────────────────────────────────────────

/// `/proc/meminfo` → totals in bytes. `MemAvailable` (not `MemFree`) is what a
/// model can actually claim, so the planner budgets against it.
pub fn mem(meminfo: &str) -> String {
    let get = |key: &str| -> u64 {
        meminfo
            .lines()
            .find(|l| l.starts_with(key) && l.as_bytes().get(key.len()) == Some(&b':'))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    let total = get("MemTotal");
    let available = get("MemAvailable");
    let swap_total = get("SwapTotal");
    let swap_free = get("SwapFree");
    format!(
        "{{\"totalB\":{},\"availableB\":{},\"usedB\":{},\"swapTotalB\":{},\"swapUsedB\":{}}}",
        total,
        available,
        total.saturating_sub(available),
        swap_total,
        swap_total.saturating_sub(swap_free)
    )
}

// ── GPU ────────────────────────────────────────────────────────────────────

/// NVIDIA: `nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,
/// memory.total,memory.used,power.draw,power.limit,compute_cap
/// --format=csv,noheader,nounits`
///
/// AMD/Intel: one tab-separated line per card, assembled by the host from
/// sysfs — `name\ttemp_mC\tbusy_pct\tvram_used_B\tvram_total_B\tpower_uW`.
///
/// A field that is `[N/A]`, empty, or unparseable becomes 0 — never an error.
/// A machine with no GPU is a supported configuration, not a failure.
pub fn gpu(nvidia_csv: &str, sysfs: &str) -> String {
    let mut out: Vec<String> = Vec::new();

    for line in nvidia_csv.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        let g = |i: usize| -> f64 { f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) };
        out.push(format!(
            concat!(
                "{{\"vendor\":\"nvidia\",\"name\":{},\"tempC\":{},\"utilPct\":{},",
                "\"vramTotalB\":{},\"vramUsedB\":{},\"powerW\":{},\"powerLimitW\":{},",
                "\"computeCap\":{}}}"
            ),
            quote(f.first().copied().unwrap_or("NVIDIA GPU")),
            num(g(1)),
            num(g(2)),
            num(g(3) * 1024.0 * 1024.0),
            num(g(4) * 1024.0 * 1024.0),
            num(g(5)),
            num(g(6)),
            num(g(7)),
        ));
    }

    for line in sysfs.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        let g = |i: usize| -> f64 {
            f.get(i)
                .and_then(|v| v.trim().parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        out.push(format!(
            concat!(
                "{{\"vendor\":\"amd\",\"name\":{},\"tempC\":{},\"utilPct\":{},",
                "\"vramTotalB\":{},\"vramUsedB\":{},\"powerW\":{},\"powerLimitW\":0,",
                "\"computeCap\":0}}"
            ),
            quote(f.first().copied().unwrap_or("GPU").trim()),
            num(g(1) / 1000.0),
            num(g(2)),
            num(g(4)),
            num(g(3)),
            num(g(5) / 1_000_000.0),
        ));
    }

    format!("[{}]", out.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPUINFO: &str = "processor\t: 0
model name\t: AMD Ryzen 9 7950X 16-Core Processor
cpu MHz\t\t: 3000.000
physical id\t: 0
core id\t\t: 0
processor\t: 1
model name\t: AMD Ryzen 9 7950X 16-Core Processor
cpu MHz\t\t: 4000.000
physical id\t: 0
core id\t\t: 0
processor\t: 2
model name\t: AMD Ryzen 9 7950X 16-Core Processor
cpu MHz\t\t: 5000.000
physical id\t: 0
core id\t\t: 1
";

    #[test]
    fn cpu_counts_physical_cores_not_threads() {
        let j = cpu(CPUINFO, "cpu  1 2 3 4\ncpu0 1 2 3 4\ncpu1 1 2 3 4\n", "");
        assert!(j.contains("\"cores\":2"), "{}", j);
        assert!(j.contains("\"threads\":3"), "{}", j);
        assert!(j.contains("\"mhz\":4000"), "{}", j);
        assert!(j.contains("7950X"));
    }

    #[test]
    fn cpu_splits_aggregate_from_per_core_stat_lines() {
        let j = cpu("", "cpu  10 0 5 85\ncpu0 1 0 1 8\ncpu1 2 0 1 7\nintr 99\n", "");
        assert!(j.contains("\"stat\":\"cpu  10 0 5 85\""), "{}", j);
        assert!(j.contains("\"coreStats\":[\"cpu0 1 0 1 8\",\"cpu1 2 0 1 7\"]"), "{}", j);
    }

    #[test]
    fn cpu_temp_takes_the_hottest_plausible_sensor() {
        let j = cpu("", "cpu  1 1 1 1", "Tctl\t45000\nTccd1\t61500\nbogus\t999000\n");
        assert!(j.contains("\"tempC\":61.5"), "{}", j);
    }

    #[test]
    fn mem_uses_available_not_free() {
        let j = mem("MemTotal:       32000 kB\nMemFree:  1000 kB\nMemAvailable:   20000 kB\nSwapTotal: 1000 kB\nSwapFree: 400 kB\n");
        assert!(j.contains("\"totalB\":32768000"), "{}", j);
        assert!(j.contains("\"availableB\":20480000"), "{}", j);
        assert!(j.contains("\"usedB\":12288000"), "{}", j);
        assert!(j.contains("\"swapUsedB\":614400"), "{}", j);
    }

    #[test]
    fn gpu_parses_nvidia_csv_into_bytes() {
        let j = gpu("NVIDIA GeForce RTX 4090, 51, 12, 24564, 1234, 62.5, 450, 8.9\n", "");
        assert!(j.contains("\"name\":\"NVIDIA GeForce RTX 4090\""), "{}", j);
        assert!(j.contains("\"vramTotalB\":25757220864"), "{}", j);
        assert!(j.contains("\"tempC\":51"), "{}", j);
        assert!(j.contains("\"powerW\":62.5"), "{}", j);
        assert!(j.contains("\"computeCap\":8.9"), "compute cap decides the build: {}", j);
    }

    #[test]
    fn gpu_tolerates_na_fields_and_no_gpu() {
        assert_eq!(gpu("", ""), "[]");
        let j = gpu("Some GPU, [N/A], [N/A], 8192, 0, [N/A], [N/A], [N/A]\n", "");
        assert!(j.contains("\"tempC\":0"), "{}", j);
        assert!(j.contains("\"vramTotalB\":8589934592"), "{}", j);
    }

    #[test]
    fn gpu_parses_sysfs_amd_line() {
        let j = gpu("", "AMD Radeon RX 7900 XTX\t45000\t33\t2147483648\t25757220864\t120000000\n");
        assert!(j.contains("\"vendor\":\"amd\""), "{}", j);
        assert!(j.contains("\"tempC\":45"), "{}", j);
        assert!(j.contains("\"vramUsedB\":2147483648"), "{}", j);
        assert!(j.contains("\"powerW\":120"), "{}", j);
    }
}
