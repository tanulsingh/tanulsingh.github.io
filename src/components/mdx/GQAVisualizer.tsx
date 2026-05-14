"use client";

import { useState } from "react";

const CORAL = "#E8976C";
const AMBER = "#D4A843";
const SAGE = "#8BAF7A";
const HC = ["#5B9BD5", "#9B59B6", "#E85B7A", "#45B7A0"];

function MatrixBlock({
  label,
  dim,
  color,
  cols = 2,
  rows = 6,
}: {
  label: string;
  dim: string;
  color: string;
  cols?: number;
  rows?: number;
}) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="font-mono text-xs font-semibold" style={{ color }}>
        {label}
      </span>
      <div
        className="rounded border"
        style={{ borderColor: color, padding: "2px", backgroundColor: `${color}08` }}
      >
        <div className="grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: "1px" }}>
          {Array.from({ length: rows * cols }).map((_, idx) => (
            <div
              key={idx}
              className="rounded-sm"
              style={{ width: "10px", height: "10px", backgroundColor: color, opacity: 0.25 }}
            />
          ))}
        </div>
      </div>
      <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
        {dim}
      </span>
    </div>
  );
}

function Times({ color = "var(--text-muted)" }: { color?: string }) {
  return (
    <span className="px-1 font-mono text-sm" style={{ color }}>
      ×
    </span>
  );
}

function Equals() {
  return (
    <span className="px-1 font-mono text-sm" style={{ color: "var(--text-muted)" }}>
      =
    </span>
  );
}

const STEPS = [
  {
    title: "The spectrum: MHA → GQA → MQA",
    subtitle: "GQA is the middle ground between full heads and full sharing",
  },
  {
    title: "GQA projections (2 groups)",
    subtitle: "Group query heads and share K, V within each group",
  },
  {
    title: "How attention works in GQA",
    subtitle: "Heads within a group share K, V but have their own Q",
  },
  {
    title: "Choosing the number of groups",
    subtitle: "The quality–memory trade-off at different group counts",
  },
];

export function GQAVisualizer() {
  const [step, setStep] = useState(0);

  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        interactive
      </p>
      <h4 className="mb-4 text-lg font-bold" style={{ fontFamily: "var(--font-serif)" }}>
        Grouped-Query Attention — Step by Step
      </h4>

      {/* Step navigation */}
      <div className="mb-5 flex items-center gap-2">
        {STEPS.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className="rounded px-2.5 py-1 font-mono text-xs transition-all"
            style={{
              backgroundColor: step === i ? "var(--tag-bg)" : "transparent",
              color: step === i ? "var(--primary)" : "var(--text-muted)",
              border: `1px solid ${step === i ? "var(--primary)" : "var(--border)"}`,
            }}
          >
            {i + 1}
          </button>
        ))}
        <span className="ml-2 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          {STEPS[step].title}
        </span>
      </div>

      {/* Step content */}
      <div className="min-h-[340px]">
        {step === 0 && <StepSpectrum />}
        {step === 1 && <StepProjections />}
        {step === 2 && <StepComputation />}
        {step === 3 && <StepGroups />}
      </div>

      {/* Navigation buttons */}
      <div className="mt-4 flex justify-between">
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="rounded px-3 py-1.5 font-mono text-xs transition-all disabled:opacity-30"
          style={{ border: "1px solid var(--border)", color: "var(--text-muted)" }}
        >
          ← prev
        </button>
        <button
          onClick={() => setStep(Math.min(STEPS.length - 1, step + 1))}
          disabled={step === STEPS.length - 1}
          className="rounded px-3 py-1.5 font-mono text-xs transition-all disabled:opacity-30"
          style={{ border: "1px solid var(--border)", color: "var(--text-muted)" }}
        >
          next →
        </button>
      </div>
    </div>
  );
}

function StepSpectrum() {
  const methods = [
    { name: "MHA", kvHeads: 4, qHeads: 4, color: "var(--text-muted)" },
    { name: "GQA (g=2)", kvHeads: 2, qHeads: 4, color: AMBER },
    { name: "MQA", kvHeads: 1, qHeads: 4, color: SAGE },
  ];

  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        GQA sits on a spectrum between MHA (every head has its own KV) and MQA (all heads share one KV).
        The parameter <strong>g</strong> (number of KV groups) controls where you land:
      </p>

      {/* Visual spectrum */}
      <div className="mb-5 flex items-center justify-between px-4">
        <div className="text-center">
          <p className="font-mono text-xs font-semibold" style={{ color: "var(--text-muted)" }}>MHA</p>
          <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>g = h</p>
        </div>
        <div className="flex-1 mx-3 h-2 rounded-full relative" style={{ backgroundColor: "var(--border)" }}>
          <div className="absolute left-0 top-0 h-full w-1/6 rounded-l-full" style={{ backgroundColor: SAGE }} />
          <div className="absolute left-1/6 top-0 h-full w-4/6" style={{ backgroundColor: AMBER, opacity: 0.6 }} />
          <div className="absolute right-0 top-0 h-full w-1/6 rounded-r-full" style={{ backgroundColor: "var(--text-muted)", opacity: 0.4 }} />
        </div>
        <div className="text-center">
          <p className="font-mono text-xs font-semibold" style={{ color: SAGE }}>MQA</p>
          <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>g = 1</p>
        </div>
      </div>

      {/* Diagram for each method showing Q heads → KV heads mapping */}
      <div className="space-y-4">
        {methods.map((m) => (
          <div key={m.name} className="rounded border p-3" style={{ borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-xs font-semibold" style={{ color: m.color }}>{m.name}</span>
              <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
                {m.kvHeads} KV head{m.kvHeads > 1 ? "s" : ""} for {m.qHeads} Q heads
              </span>
            </div>
            <div className="flex items-center gap-4">
              {/* Q heads */}
              <div className="flex flex-col items-center gap-1">
                <span className="font-mono" style={{ fontSize: "9px", color: CORAL }}>Q heads</span>
                <div className="flex gap-1">
                  {Array.from({ length: m.qHeads }).map((_, i) => (
                    <div
                      key={i}
                      className="rounded"
                      style={{
                        width: "24px",
                        height: "24px",
                        backgroundColor: HC[i],
                        opacity: 0.6,
                        border: `1.5px solid ${HC[i]}`,
                      }}
                    />
                  ))}
                </div>
              </div>

              {/* Arrow */}
              <svg width="40" height="24" viewBox="0 0 40 24">
                <path d="M0 12 L30 12 M24 6 L34 12 L24 18" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" />
              </svg>

              {/* KV heads with grouping indicators */}
              <div className="flex flex-col items-center gap-1">
                <span className="font-mono" style={{ fontSize: "9px", color: AMBER }}>KV heads</span>
                <div className="flex gap-1">
                  {Array.from({ length: m.kvHeads }).map((_, i) => {
                    const groupSize = m.qHeads / m.kvHeads;
                    const startHead = i * groupSize;
                    return (
                      <div key={i} className="flex flex-col items-center gap-0.5">
                        <div
                          className="rounded"
                          style={{
                            width: `${groupSize * 24 + (groupSize - 1) * 4}px`,
                            height: "24px",
                            backgroundColor: AMBER,
                            opacity: 0.3,
                            border: `1.5px solid ${AMBER}`,
                          }}
                        />
                        <span className="font-mono" style={{ fontSize: "8px", color: "var(--text-muted)" }}>
                          serves Q{startHead + 1}{groupSize > 1 ? `-Q${startHead + groupSize}` : ""}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StepProjections() {
  const groupColors = [HC[0], HC[2]]; // blue group, pink group

  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        With <strong>4 query heads</strong> and <strong>g=2 KV groups</strong>: heads 1-2 share one KV,
        heads 3-4 share another. Let&apos;s see the exact weight matrix shapes (d<sub>model</sub>=8, d<sub>k</sub>=2):
      </p>

      {/* Q projections — still 4 separate W_Q */}
      <div className="mb-5">
        <p className="mb-2 font-mono text-xs" style={{ color: CORAL }}>
          Query: 4 separate projections (unchanged from MHA)
        </p>
        <div className="flex items-center gap-1 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" cols={8} rows={6} />
          <Times color={CORAL} />
          <div className="flex gap-0.5">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`W_Q${i + 1}`} dim="8×2" color={c} cols={2} rows={8} />
            ))}
          </div>
          <Equals />
          <div className="flex gap-0.5">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`Q${i + 1}`} dim="6×2" color={c} cols={2} rows={6} />
            ))}
          </div>
        </div>
        <p className="mt-1 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
          Total Q params: 4 × (8×2) = 64
        </p>
      </div>

      {/* K projections — 2 groups */}
      <div className="mb-5">
        <p className="mb-2 font-mono text-xs" style={{ color: AMBER }}>
          Key: only 2 projections (one per group)
        </p>
        <div className="flex items-center gap-1 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" cols={8} rows={6} />
          <Times color={AMBER} />
          <div className="flex gap-2">
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[0], backgroundColor: `${groupColors[0]}06` }}>
              <MatrixBlock label="W_K1" dim="8×2" color={groupColors[0]} cols={2} rows={8} />
              <p className="mt-0.5 text-center font-mono" style={{ fontSize: "7px", color: groupColors[0] }}>
                → Q1, Q2
              </p>
            </div>
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[1], backgroundColor: `${groupColors[1]}06` }}>
              <MatrixBlock label="W_K2" dim="8×2" color={groupColors[1]} cols={2} rows={8} />
              <p className="mt-0.5 text-center font-mono" style={{ fontSize: "7px", color: groupColors[1] }}>
                → Q3, Q4
              </p>
            </div>
          </div>
          <Equals />
          <div className="flex gap-2">
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[0], backgroundColor: `${groupColors[0]}06` }}>
              <MatrixBlock label="K_g1" dim="6×2" color={groupColors[0]} cols={2} rows={6} />
            </div>
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[1], backgroundColor: `${groupColors[1]}06` }}>
              <MatrixBlock label="K_g2" dim="6×2" color={groupColors[1]} cols={2} rows={6} />
            </div>
          </div>
        </div>
        <p className="mt-1 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
          Total K params: 2 × (8×2) = 32 <span style={{ color: CORAL }}>(vs 64 in MHA — 2× fewer)</span>
        </p>
      </div>

      {/* V projections — 2 groups */}
      <div className="mb-5">
        <p className="mb-2 font-mono text-xs" style={{ color: SAGE }}>
          Value: only 2 projections (same grouping as K)
        </p>
        <div className="flex items-center gap-1 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" cols={8} rows={6} />
          <Times color={SAGE} />
          <div className="flex gap-2">
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[0], backgroundColor: `${groupColors[0]}06` }}>
              <MatrixBlock label="W_V1" dim="8×2" color={groupColors[0]} cols={2} rows={8} />
              <p className="mt-0.5 text-center font-mono" style={{ fontSize: "7px", color: groupColors[0] }}>
                → Q1, Q2
              </p>
            </div>
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[1], backgroundColor: `${groupColors[1]}06` }}>
              <MatrixBlock label="W_V2" dim="8×2" color={groupColors[1]} cols={2} rows={8} />
              <p className="mt-0.5 text-center font-mono" style={{ fontSize: "7px", color: groupColors[1] }}>
                → Q3, Q4
              </p>
            </div>
          </div>
          <Equals />
          <div className="flex gap-2">
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[0], backgroundColor: `${groupColors[0]}06` }}>
              <MatrixBlock label="V_g1" dim="6×2" color={groupColors[0]} cols={2} rows={6} />
            </div>
            <div className="rounded border p-1.5" style={{ borderColor: groupColors[1], backgroundColor: `${groupColors[1]}06` }}>
              <MatrixBlock label="V_g2" dim="6×2" color={groupColors[1]} cols={2} rows={6} />
            </div>
          </div>
        </div>
        <p className="mt-1 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
          Total V params: 2 × (8×2) = 32 <span style={{ color: CORAL }}>(vs 64 in MHA — 2× fewer)</span>
        </p>
      </div>

      {/* Implementation hint */}
      <div
        className="rounded p-3"
        style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
      >
        <p className="font-mono text-xs mb-1" style={{ color: "var(--text-muted)" }}>
          <strong>In code:</strong> you can implement this as a single W<sub>K</sub> of shape (d<sub>model</sub> × g×d<sub>k</sub>) = (8×4),
          then reshape the output to (batch, seq, g, d<sub>k</sub>) and broadcast to match the query heads.
        </p>
        <p className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          <strong>KV cache per token:</strong> 2 K + 2 V = 4 vectors of dim 2 = <span style={{ color: AMBER }}>8 values</span>
          &nbsp;(MHA: 16 · MQA: 4)
        </p>
      </div>
    </div>
  );
}

function StepComputation() {
  const groupColors = [HC[0], HC[2]];

  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        Heads within the same group share K and V but compute independently with their own Q.
        Heads in <em>different</em> groups see <em>different</em> K and V — more diversity than MQA.
      </p>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {/* Group 1 */}
        <div className="rounded border p-3" style={{ borderColor: groupColors[0], backgroundColor: `${groupColors[0]}06` }}>
          <p className="mb-2 font-mono text-xs font-semibold" style={{ color: groupColors[0] }}>
            Group 1 — shared K<sub>g1</sub>, V<sub>g1</sub>
          </p>
          {[0, 1].map((i) => (
            <div key={i} className="mb-2 flex items-center gap-1 flex-wrap">
              <MatrixBlock label={`Q${i + 1}`} dim="6×2" color={HC[i]} cols={2} rows={4} />
              <Times color={HC[i]} />
              <MatrixBlock label="K_g1^T" dim="2×6" color={groupColors[0]} cols={4} rows={2} />
              <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>→ out{i + 1}</span>
            </div>
          ))}
        </div>

        {/* Group 2 */}
        <div className="rounded border p-3" style={{ borderColor: groupColors[1], backgroundColor: `${groupColors[1]}06` }}>
          <p className="mb-2 font-mono text-xs font-semibold" style={{ color: groupColors[1] }}>
            Group 2 — shared K<sub>g2</sub>, V<sub>g2</sub>
          </p>
          {[2, 3].map((i) => (
            <div key={i} className="mb-2 flex items-center gap-1 flex-wrap">
              <MatrixBlock label={`Q${i + 1}`} dim="6×2" color={HC[i]} cols={2} rows={4} />
              <Times color={HC[i]} />
              <MatrixBlock label="K_g2^T" dim="2×6" color={groupColors[1]} cols={4} rows={2} />
              <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>→ out{i + 1}</span>
            </div>
          ))}
        </div>
      </div>

      <div
        className="rounded p-3 mt-4"
        style={{ borderLeft: `3px solid ${AMBER}`, backgroundColor: `${AMBER}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>Key difference from MQA:</strong> Group 1 and Group 2 have <em>different</em> learned K and V projections.
          Q1 and Q2 share one &quot;information store,&quot; while Q3 and Q4 share a different one.
          This preserves more diversity than MQA while still halving the KV cache vs. MHA.
        </p>
      </div>
    </div>
  );
}

function StepGroups() {
  const [selectedG, setSelectedG] = useState(2);
  const totalHeads = 32;
  const dk = 128;
  const layers = 80;
  const seqLen = 32768;

  const configs = [
    { g: 32, label: "MHA (g=32)", color: "var(--text-muted)" },
    { g: 8, label: "GQA (g=8)", color: AMBER },
    { g: 4, label: "GQA (g=4)", color: AMBER },
    { g: 2, label: "GQA (g=2)", color: AMBER },
    { g: 1, label: "MQA (g=1)", color: SAGE },
  ];

  const mhaCache = 2 * totalHeads * seqLen * dk * layers * 2 / 1e9;

  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        The number of groups <strong>g</strong> controls the trade-off. Click to explore different configurations
        (32 query heads, d<sub>k</sub>=128, 80 layers, 32K sequence):
      </p>

      {/* Interactive group selector */}
      <div className="mb-4 flex flex-wrap gap-2">
        {configs.map((c) => (
          <button
            key={c.g}
            onClick={() => setSelectedG(c.g)}
            className="rounded px-3 py-1.5 font-mono text-xs transition-all"
            style={{
              backgroundColor: selectedG === c.g ? "var(--tag-bg)" : "transparent",
              color: selectedG === c.g ? "var(--primary)" : "var(--text-muted)",
              border: `1px solid ${selectedG === c.g ? "var(--primary)" : "var(--border)"}`,
            }}
          >
            {c.label}
          </button>
        ))}
      </div>

      {/* Visual: Q heads grouped */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          Head grouping (showing 8 of 32 query heads):
        </p>
        <div className="flex gap-1 flex-wrap">
          {Array.from({ length: 8 }).map((_, i) => {
            const groupIdx = Math.floor(i / (8 / Math.min(selectedG, 8)));
            const groupColor = HC[groupIdx % 4];
            return (
              <div key={i} className="flex flex-col items-center">
                <div
                  className="rounded"
                  style={{
                    width: "28px",
                    height: "28px",
                    backgroundColor: groupColor,
                    opacity: 0.5,
                    border: `2px solid ${groupColor}`,
                  }}
                />
                <span className="font-mono" style={{ fontSize: "8px", color: "var(--text-muted)" }}>Q{i + 1}</span>
              </div>
            );
          })}
        </div>
        <p className="mt-1 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
          same color = same KV group (shares K and V)
        </p>
      </div>

      {/* Stats for selected config */}
      <div className="rounded border p-4" style={{ borderColor: "var(--border)", backgroundColor: "var(--bg)" }}>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>KV heads</p>
            <p className="font-mono text-lg font-bold" style={{ color: AMBER }}>{selectedG}</p>
          </div>
          <div>
            <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>Q heads per KV</p>
            <p className="font-mono text-lg font-bold" style={{ color: CORAL }}>{totalHeads / selectedG}</p>
          </div>
          <div>
            <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>KV cache (32K seq)</p>
            <p className="font-mono text-lg font-bold" style={{ color: SAGE }}>
              {(2 * selectedG * seqLen * dk * layers * 2 / 1e9).toFixed(1)} GB
            </p>
          </div>
          <div>
            <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>Savings vs MHA</p>
            <p className="font-mono text-lg font-bold" style={{ color: CORAL }}>
              {(totalHeads / selectedG)}×
            </p>
          </div>
        </div>

        {/* Bar visualization */}
        <div className="mt-4">
          <div className="flex items-center gap-2">
            <span className="w-16 shrink-0 font-mono text-xs text-right" style={{ color: "var(--text-muted)" }}>
              MHA
            </span>
            <div className="flex-1 h-4 rounded overflow-hidden" style={{ backgroundColor: "var(--bg-surface)" }}>
              <div className="h-full rounded" style={{ width: "100%", backgroundColor: "var(--text-muted)", opacity: 0.3 }} />
            </div>
            <span className="w-14 shrink-0 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
              {mhaCache.toFixed(1)} GB
            </span>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span className="w-16 shrink-0 font-mono text-xs text-right" style={{ color: AMBER }}>
              g={selectedG}
            </span>
            <div className="flex-1 h-4 rounded overflow-hidden" style={{ backgroundColor: "var(--bg-surface)" }}>
              <div
                className="h-full rounded transition-all duration-300"
                style={{
                  width: `${(selectedG / totalHeads) * 100}%`,
                  backgroundColor: selectedG === 1 ? SAGE : AMBER,
                  opacity: 0.6,
                }}
              />
            </div>
            <span className="w-14 shrink-0 font-mono text-xs" style={{ color: AMBER }}>
              {(2 * selectedG * seqLen * dk * layers * 2 / 1e9).toFixed(1)} GB
            </span>
          </div>
        </div>
      </div>

      {/* Quality note */}
      <div
        className="rounded p-3 mt-4"
        style={{ borderLeft: `3px solid ${SAGE}`, backgroundColor: `${SAGE}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>The sweet spot:</strong> Llama 3 uses g=8 (8 KV heads for 64 query heads). This gives 8× cache
          reduction with virtually no quality loss. Going below g=4 starts to show measurable degradation
          on reasoning benchmarks.
        </p>
      </div>
    </div>
  );
}
