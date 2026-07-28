// src/ui/ModelsPanel.tsx — the model library.
//
// One button finds every GGUF on the machine; the table shows what each one
// actually is (architecture, quantisation, layers, trained context, MoE width)
// because those are the facts that decide whether it will run here — the file
// name is only a rumour about them.

import { models } from "../cell/models.ts";
import { cfg } from "../cell/cfg.ts";
import { plan as computePlan } from "../lib/plan.ts";
import { bytes, shortPath, stamp } from "../lib/format.ts";
import { runModel } from "./actions.ts";
import { Bar, Empty, ErrorNote, KV, Panel, Pill } from "./kit.tsx";
import { MemoryPlan } from "./Memory.tsx";
import {
  currentModel,
  modelsSizeB,
  planningHw,
  visibleModels,
} from "./derive.ts";

function Dirs() {
  return (
    <Panel title="Search paths" icon="⌸">
      <ul class="dirs">
        {models.dirs.length === 0
          ? <li class="dim">Default locations are used until you add one.</li>
          : models.dirs.map((d) => (
            <li key={d}>
              <span class="mono" title={d}>{shortPath(d, 56)}</span>
              <button
                type="button"
                class="btn tiny"
                title={`Stop searching ${d}`}
                onClick={() => models.removeDir(d)}
              >
                ✕
              </button>
            </li>
          ))}
      </ul>
      <form
        class="add-dir"
        onSubmit={(e) => {
          e.preventDefault();
          const input = (e.currentTarget as HTMLFormElement)
            .elements.namedItem("dir") as HTMLInputElement | null;
          if (input?.value) {
            models.addDir(input.value);
            input.value = "";
            models.scan();
          }
        }}
      >
        <input
          name="dir"
          placeholder="Add a directory"
          aria-label="Add a directory"
        />
        <button type="submit" class="btn small">Add</button>
      </form>
    </Panel>
  );
}

function Details() {
  const m = currentModel();
  if (!m) {
    return (
      <Panel title="Model" icon="◈">
        <Empty
          icon="◇"
          title="No model selected"
          hint="Pick one from the table."
        />
      </Panel>
    );
  }
  const meta = m.meta;
  return (
    <Panel
      title="Model"
      icon="◈"
      right={meta
        ? <Pill tone="accent">{meta.quant}</Pill>
        : <Pill tone="bad">unreadable</Pill>}
    >
      <div class="model-name" title={m.path}>{m.file}</div>
      {m.metaError
        ? <ErrorNote message={`Header could not be read: ${m.metaError}`} />
        : null}
      {meta
        ? (
          <>
            <div class="kv-grid">
              <KV k="Architecture" v={meta.arch} />
              <KV k="Name" v={meta.name || "—"} />
              <KV k="Layers" v={String(meta.nLayer)} />
              <KV k="Trained context" v={meta.nCtxTrain.toLocaleString()} />
              <KV k="Embedding" v={String(meta.nEmbd)} />
              <KV k="Heads" v={`${meta.nHead} · ${meta.nHeadKv} kv`} />
              <KV
                k="Experts"
                v={meta.nExpert > 0
                  ? `${meta.nExpert} (${meta.nExpertUsed} active)`
                  : "dense"}
                tip="Mixture-of-experts models can keep their experts in RAM while attention stays on the GPU"
              />
              <KV k="Tensors" v={String(meta.nTensors)} />
              <KV k="Weights" v={bytes(meta.tensorBytes)} mono />
              <KV k="File" v={bytes(m.sizeB)} mono />
            </div>
            {meta.unknownTypes > 0
              ? (
                <div class="warn-note">
                  {meta.unknownTypes}{" "}
                  tensor(s) use a ggml type this build does not know — the sizes
                  above exclude them.
                </div>
              )
              : null}
            <MemoryPlan
              plan={computePlan(meta, planningHw(), cfg.settings)}
              compact
            />
          </>
        )
        : null}
    </Panel>
  );
}

export function ModelsPanel() {
  const list = visibleModels();
  const p = models.progress;
  return (
    <div class="tab-body">
      <ErrorNote message={models.lastError} />
      <div class="cols">
        <Panel
          title="Detected models"
          icon="◈"
          wide
          right={
            <>
              <input
                class="filter"
                placeholder="Filter"
                aria-label="Filter"
                value={models.filter}
                onInput={(e) =>
                  models.setFilter((e.currentTarget as HTMLInputElement).value)}
              />
              <Pill tone="idle">{bytes(modelsSizeB())}</Pill>
              <button
                type="button"
                class="btn primary small"
                t="detect-models"
                onClick={() => models.scan()}
                disabled={models.scanning}
              >
                {models.scanning ? "Scanning…" : "Detect models"}
              </button>
            </>
          }
        >
          {p
            ? (
              <div class="scan-progress">
                <div class="sub-label">
                  Reading headers {p.done}/{p.total} · {p.current}
                </div>
                <Bar
                  value={p.done}
                  max={Math.max(1, p.total)}
                  tone="busy"
                  height={6}
                />
              </div>
            )
            : null}
          {list.length === 0 && !models.scanning
            ? (
              <Empty
                icon="◇"
                title={models.lastScan
                  ? "No GGUF files found"
                  : "Nothing scanned yet"}
                hint="Press “Detect models”. Add a directory on the right if your models live somewhere unusual."
              />
            )
            : (
              <table class="table dense" t="models-table">
                <thead>
                  <tr>
                    <th />
                    <th>Model</th>
                    <th>From</th>
                    <th>Arch</th>
                    <th>Quant</th>
                    <th>Layers</th>
                    <th>Ctx</th>
                    <th>Size</th>
                    <th>Found</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {list.map((m) => (
                    <tr
                      key={m.path}
                      class={m.path === models.selected ? "row-active" : ""}
                      onClick={() => models.select(m.path)}
                    >
                      <td class="c-icon">
                        <input
                          type="radio"
                          aria-label={`Select ${m.file}`}
                          checked={m.path === models.selected}
                          onChange={() => models.select(m.path)}
                        />
                      </td>
                      <td class="c-file" title={m.path}>{m.file}</td>
                      <td>
                        <Pill
                          tone={m.source === "file" ? "idle" : "accent"}
                          title={m.dir}
                        >
                          {m.source === "file" ? "disk" : m.source}
                        </Pill>
                      </td>
                      <td>{m.meta?.arch ?? "—"}</td>
                      <td>
                        {m.meta
                          ? <Pill tone="accent">{m.meta.quant}</Pill>
                          : <Pill tone="bad">?</Pill>}
                      </td>
                      <td class="mono">{m.meta?.nLayer ?? "—"}</td>
                      <td class="mono">
                        {m.meta?.nCtxTrain?.toLocaleString() ?? "—"}
                      </td>
                      <td class="mono">{bytes(m.sizeB)}</td>
                      <td class="dim">{stamp(m.mtime)}</td>
                      <td class="c-act">
                        <button
                          type="button"
                          class="btn tiny primary"
                          title="Select, tune for this machine, and start the server"
                          onClick={(e) => {
                            e.stopPropagation();
                            runModel(m.path);
                          }}
                        >
                          Run
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        </Panel>
        <Details />
        <Dirs />
      </div>
    </div>
  );
}
