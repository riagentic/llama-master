// src/ui/Memory.tsx — "where does this model go", drawn.
//
// The kata asks for a visual representation of VRAM and RAM with the model
// placed in them per configuration. This is that picture: two capacity bars,
// each split into the things that actually claim the memory, plus the layer
// strip that shows which transformer blocks landed on which device.

// `useLocal` for hover state rather than a cell: it is per-component-instance
// view state that nothing else reads (dep/aio/docs/ui/air-reference.md).
import { useLocal } from "aio/air";
import type { Plan, Pool } from "../lib/plan.ts";
import type { Segment } from "./kit.tsx";
import { Bar, KV, Legend, StackBar } from "./kit.tsx";
import { bytes, pctLabel } from "../lib/format.ts";

function segments(pool: Plan["vram"]): Segment[] {
  return pool.buckets.map((b) => ({
    key: b.key,
    label: b.label,
    bytes: b.bytes,
    tone: b.key,
  }));
}

function PoolBar(props: { pool: Plan["vram"]; hint?: string }) {
  const p = props.pool;
  const segs = segments(p);
  const withOther: Segment[] = p.otherB > 0
    ? [...segs, {
      key: "other",
      label: "In use elsewhere",
      bytes: p.otherB,
      tone: "other",
    }]
    : segs;
  const claimed = p.usedB + p.otherB;
  return (
    <div class="pool">
      <div class="pool-head">
        <span class="pool-name">{p.label}</span>
        <span class={p.overB > 0 ? "pool-total over" : "pool-total"}>
          {bytes(claimed)} / {bytes(p.capacityB)}
          <em>{pctLabel(claimed, p.capacityB)}</em>
        </span>
      </div>
      <StackBar
        capacityB={p.capacityB}
        segments={withOther}
        overB={p.overB}
        height={16}
      />
      <Legend segments={withOther} />
      {p.overB > 0
        ? (
          <div class="pool-over" t="over">
            Over by {bytes(p.overB)}
          </div>
        )
        : (
          <div class="pool-free">
            {bytes(p.freeB)} free{props.hint ? ` · ${props.hint}` : ""}
          </div>
        )}
    </div>
  );
}

/** One cell per transformer layer: GPU, GPU-with-experts-in-RAM, or CPU. */
function LayerStrip(props: { plan: Plan }) {
  const p = props.plan;
  if (p.nLayer === 0) return null;
  const firstGpu = p.nLayer - p.layersOnGpu;
  const cells = [];
  for (let i = 0; i < p.nLayer; i++) {
    const onGpu = i >= firstGpu;
    const moe = i < p.moeOnCpu;
    const cls = !onGpu ? "lay cpu" : moe ? "lay split" : "lay gpu";
    const title = !onGpu
      ? `Layer ${i}: CPU`
      : moe
      ? `Layer ${i}: attention on GPU, experts in RAM`
      : `Layer ${i}: GPU`;
    cells.push(<i key={String(i)} class={cls} title={title} />);
  }
  return (
    <div class="layers">
      <div class="layers-strip">{cells}</div>
      <div class="layers-key">
        <span>
          <i class="lay gpu" />GPU {p.layersOnGpu}
        </span>
        {p.moeOnCpu > 0
          ? (
            <span>
              <i class="lay split" />experts in RAM {p.moeOnCpu}
            </span>
          )
          : null}
        <span>
          <i class="lay cpu" />CPU {p.nLayer - p.layersOnGpu}
        </span>
      </div>
    </div>
  );
}

// ── the interactive memory map ─────────────────────────────────────────────

/** One band of the map: a slice of a pool, drawn to scale. */
type Band = { key: string; label: string; bytes: number; detail: string };

function bands(pool: Pool, kind: "vram" | "ram"): Band[] {
  const out: Band[] = pool.buckets.map((b) => ({
    key: b.key,
    label: b.label,
    bytes: b.bytes,
    detail: `${b.label} — ${bytes(b.bytes)} of ${kind.toUpperCase()}`,
  }));
  if (pool.otherB > 0) {
    out.push({
      key: "other",
      label: kind === "vram" ? "Other GPU users" : "Rest of the system",
      bytes: pool.otherB,
      detail: `${bytes(pool.otherB)} already in use by something else`,
    });
  }
  return out.filter((b) => b.bytes > 0);
}

function Region(props: {
  pool: Pool;
  kind: "vram" | "ram";
  /** Region caption; defaults to the kind. Per-card regions name the card. */
  title?: string;
  tip?: string;
  onHover: (detail: string) => void;
}) {
  const p = props.pool;
  const claimed = p.usedB + p.otherB;
  return (
    <div
      class={`map-region region-${props.kind}`}
      style={{ flexGrow: String(Math.max(1, p.capacityB)) }}
      onMouseLeave={() => props.onHover("")}
    >
      <div class="map-region-head" title={props.tip}>
        <b>{props.title ?? props.kind.toUpperCase()}</b>
        <span>{bytes(p.capacityB)}</span>
      </div>
      <div class="map-region-body">
        {
          /* The semi-opaque layer: what this configuration will actually take,
            laid over the capacity it is taking it from. */
        }
        <div class="map-fill">
          {bands(p, props.kind).map((b) => (
            <div
              key={b.key}
              class={`map-band seg-${b.key}`}
              style={{
                width: `${
                  ((b.bytes / Math.max(1, p.capacityB)) * 100).toFixed(3)
                }%`,
              }}
              title={b.detail}
              onMouseEnter={() => props.onHover(b.detail)}
            />
          ))}
          {p.overB > 0
            ? (
              <div
                class="map-band seg-over"
                style={{ width: "100%" }}
                title={`Over capacity by ${bytes(p.overB)}`}
                onMouseEnter={() =>
                  props.onHover(
                    `Does not fit — ${
                      bytes(p.overB)
                    } more than this ${props.kind.toUpperCase()} has`,
                  )}
              />
            )
            : null}
        </div>
      </div>
      <div class="map-region-foot">
        {pctLabel(claimed, p.capacityB)} used · {bytes(p.freeB)} free
      </div>
    </div>
  );
}

/** A card of the device plan, shaped as the Pool the map's Region draws —
 *  one implementation of the band-drawing, whatever it is drawing. */
function cardAsPool(c: Plan["devices"]["cards"][number], i: number): Pool {
  const usedB = c.weightsB + c.kvB + c.computeB;
  return {
    label: `GPU ${i}`,
    capacityB: c.capacityB,
    usedB,
    otherB: c.otherB,
    freeB: Math.max(0, c.capacityB - usedB - c.otherB),
    overB: c.overB,
    buckets: [
      { key: "weights", label: "Weights", bytes: c.weightsB },
      { key: "kv", label: "KV cache", bytes: c.kvB },
      { key: "compute", label: "Compute (est.)", bytes: c.computeB },
    ],
  };
}

/**
 * Every byte of memory on the machine, to scale, with this configuration laid
 * over it.
 *
 * The pools are drawn side by side and proportional to their real capacities —
 * and a machine with two cards shows TWO regions, not one pooled "VRAM": a
 * second GPU is not a bigger GPU, llama.cpp places layers per card, and the
 * question a two-card machine actually asks is "which card is full, with
 * what". One pooled bar could read "fits" while a card overflowed.
 */
export function MemoryMap(props: { plan: Plan }) {
  const [hovered, setHovered] = useLocal("");
  const p = props.plan;
  const total = p.vram.capacityB + p.ram.capacityB;
  const perCard = p.devices.cards.length > 1;
  return (
    <div class="memmap" t="memory-map">
      <div class="map-head">
        <span class="pool-name">Memory map</span>
        <span class="dim">{bytes(total)} total on this machine</span>
      </div>
      <div class="map-track">
        {perCard
          ? p.devices.cards.map((c, i) => (
            <Region
              key={c.name + String(i)}
              pool={cardAsPool(c, i)}
              kind="vram"
              title={`GPU ${i}`}
              tip={c.name}
              onHover={setHovered}
            />
          ))
          : <Region pool={p.vram} kind="vram" onHover={setHovered} />}
        <Region pool={p.ram} kind="ram" onHover={setHovered} />
      </div>
      {p.devices.unplacedB > 0
        ? (
          <div class="map-unplaced" t="map-unplaced">
            {bytes(p.devices.unplacedB)}{" "}
            of layers have nowhere to go — no card has room for them, however
            the cut is made.
          </div>
        )
        : null}
      <div class="map-legend">
        <span class="legend-item">
          <i class="legend-dot region-swatch-vram" />VRAM
        </span>
        <span class="legend-item">
          <i class="legend-dot region-swatch-ram" />RAM
        </span>
        <span class="legend-item">
          <i class="legend-dot seg-weights" />Weights
        </span>
        <span class="legend-item">
          <i class="legend-dot seg-experts" />Experts
        </span>
        <span class="legend-item">
          <i class="legend-dot seg-kv" />KV cache
        </span>
        <span class="legend-item">
          <i class="legend-dot seg-compute" />Compute (est.)
        </span>
        <span class="legend-item">
          <i class="legend-dot seg-other" />In use elsewhere
        </span>
      </div>
      <div class="map-hover" t="map-hover">
        {hovered || "Hover a band to see what claims it."}
      </div>
    </div>
  );
}

export function MemoryPlan(props: { plan: Plan; compact?: boolean }) {
  const p = props.plan;
  return (
    <div class="memplan" t="memory-plan">
      {props.compact ? null : <MemoryMap plan={p} />}
      <PoolBar pool={p.vram} />
      <PoolBar pool={p.ram} />
      {props.compact ? null : <LayerStrip plan={p} />}
      {props.compact ? null : (
        <div class="memfacts">
          <KV
            k="Context"
            v={`${p.ctx.toLocaleString()} tokens`}
            tip="Tokens the KV cache is sized for"
          />
          <KV
            k="KV cache"
            v={`${bytes(p.kvTotalB)} · ${bytes(p.kvPerTokenB * 1024)}/1k tok`}
            tip="Grows linearly with context — the first thing to trim"
          />
          <KV k="Layers on GPU" v={`${p.layersOnGpu} / ${p.nLayer}`} />
          <KV
            k="Verdict"
            v={p.fits
              ? <span class="ok-text">fits</span>
              : <span class="bad-text">does not fit</span>}
          />
        </div>
      )}
      {p.notes.length > 0
        ? (
          <ul class="notes" t="plan-notes">
            {p.notes.map((n) => <li key={n}>{n}</li>)}
          </ul>
        )
        : null}
    </div>
  );
}

/** The header strip: two thin bars that are always on screen, whatever tab the
 *  user is on. Live pressure, not a plan. */
export function MemoryMini(props: {
  vramUsedB: number;
  vramTotalB: number;
  ramUsedB: number;
  ramTotalB: number;
}) {
  return (
    <div class="memmini">
      <div class="memmini-row">
        <span>VRAM</span>
        <Bar
          value={props.vramUsedB}
          max={props.vramTotalB}
          tone={props.vramUsedB / Math.max(1, props.vramTotalB) > 0.9
            ? "bad"
            : "accent"}
          height={4}
        />
        <b>{bytes(props.vramUsedB)}</b>
      </div>
      <div class="memmini-row">
        <span>RAM</span>
        <Bar
          value={props.ramUsedB}
          max={props.ramTotalB}
          tone={props.ramUsedB / Math.max(1, props.ramTotalB) > 0.9
            ? "bad"
            : "ok"}
          height={4}
        />
        <b>{bytes(props.ramUsedB)}</b>
      </div>
    </div>
  );
}
