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
  width,
  height,
  headColors,
}: {
  label: string;
  dim: string;
  color: string;
  width: number;
  height: number;
  headColors?: string[];
}) {
  const cols = headColors ? headColors.length * 2 : Math.round(width / 12);
  const rows = Math.round(height / 12);

  return (
    <div className="flex flex-col items-center gap-1">
      <span className="font-mono text-xs font-semibold" style={{ color }}>
        {label}
      </span>
      <div
        className="rounded border"
        style={{
          borderColor: color,
          padding: "2px",
          backgroundColor: `${color}08`,
        }}
      >
        <div className="grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: "1px" }}>
          {Array.from({ length: rows * cols }).map((_, idx) => {
            const col = idx % cols;
            const cellColor = headColors
              ? headColors[Math.floor(col / 2)]
              : color;
            return (
              <div
                key={idx}
                className="rounded-sm"
                style={{
                  width: "10px",
                  height: "10px",
                  backgroundColor: cellColor,
                  opacity: 0.25,
                }}
              />
            );
          })}
        </div>
      </div>
      <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
        {dim}
      </span>
    </div>
  );
}

function Arrow({ color = "var(--text-muted)" }: { color?: string }) {
  return (
    <div className="flex items-center px-1">
      <svg width="24" height="12" viewBox="0 0 24 12">
        <path d="M0 6 L18 6 M14 2 L20 6 L14 10" fill="none" stroke={color} strokeWidth="1.5" />
      </svg>
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
    title: "Multi-Head Attention (baseline)",
    subtitle: "Each head has its own Q, K, V projections",
  },
  {
    title: "Multi-Query Attention (the change)",
    subtitle: "K and V collapse to a single shared head",
  },
  {
    title: "How attention works in MQA",
    subtitle: "Each query head attends against the same K, V",
  },
  {
    title: "The memory savings",
    subtitle: "KV cache comparison at 32K sequence length",
  },
];

export function MQAVisualizer() {
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
        Multi-Query Attention — Step by Step
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
      <div className="min-h-[320px]">
        {step === 0 && <StepMHA />}
        {step === 1 && <StepMQA />}
        {step === 2 && <StepComputation />}
        {step === 3 && <StepMemory />}
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

function StepMHA() {
  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        In standard Multi-Head Attention, each of the <strong>4 heads</strong> has its own
        separate W<sub>Q</sub>, W<sub>K</sub>, and W<sub>V</sub> projection. For our example
        (d<sub>model</sub>=8, 4 heads, d<sub>k</sub>=2):
      </p>

      {/* Q projections — 4 separate */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: CORAL }}>
          Query projections (4 separate):
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" width={96} height={72} />
          <Times color={CORAL} />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`W_Q${i + 1}`} dim="8×2" color={c} width={24} height={96} />
            ))}
          </div>
          <Equals />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`Q${i + 1}`} dim="6×2" color={c} width={24} height={72} />
            ))}
          </div>
        </div>
      </div>

      {/* K projections — 4 separate */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: AMBER }}>
          Key projections (4 separate):
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" width={96} height={72} />
          <Times color={AMBER} />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`W_K${i + 1}`} dim="8×2" color={c} width={24} height={96} />
            ))}
          </div>
          <Equals />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`K${i + 1}`} dim="6×2" color={c} width={24} height={72} />
            ))}
          </div>
        </div>
      </div>

      {/* V projections — 4 separate */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: SAGE }}>
          Value projections (4 separate):
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" width={96} height={72} />
          <Times color={SAGE} />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`W_V${i + 1}`} dim="8×2" color={c} width={24} height={96} />
            ))}
          </div>
          <Equals />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`V${i + 1}`} dim="6×2" color={c} width={24} height={72} />
            ))}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div
        className="rounded p-3 mt-4"
        style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          <strong>KV cache per token:</strong> 4 K heads + 4 V heads = 8 vectors of size 2 = <span style={{ color: CORAL }}>16 values/token</span>
        </p>
        <p className="font-mono text-xs mt-1" style={{ color: "var(--text-muted)" }}>
          <strong>KV parameters:</strong> 4×(8×2) + 4×(8×2) = <span style={{ color: CORAL }}>128 parameters</span>
        </p>
      </div>
    </div>
  );
}

function StepMQA() {
  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        In MQA, query projections stay the same (4 separate heads), but K and V collapse to a <strong>single shared projection</strong>.
        The key insight: we only need one set of keys and values — the diversity comes from the queries.
      </p>

      {/* Q projections — still 4 separate */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: CORAL }}>
          Query projections (still 4 separate — unchanged):
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" width={96} height={72} />
          <Times color={CORAL} />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`W_Q${i + 1}`} dim="8×2" color={c} width={24} height={96} />
            ))}
          </div>
          <Equals />
          <div className="flex gap-1">
            {HC.map((c, i) => (
              <MatrixBlock key={i} label={`Q${i + 1}`} dim="6×2" color={c} width={24} height={72} />
            ))}
          </div>
        </div>
      </div>

      {/* K projection — single */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: AMBER }}>
          Key projection (<span style={{ color: CORAL }}>single shared</span> — this is the change!):
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" width={96} height={72} />
          <Times color={AMBER} />
          <MatrixBlock label="W_K" dim="8×2" color={AMBER} width={24} height={96} />
          <Equals />
          <MatrixBlock label="K" dim="6×2" color={AMBER} width={24} height={72} />
          <span className="ml-2 rounded px-2 py-0.5 font-mono text-xs" style={{ backgroundColor: `${CORAL}20`, color: CORAL, border: `1px solid ${CORAL}40` }}>
            shared by all 4 heads
          </span>
        </div>
      </div>

      {/* V projection — single */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: SAGE }}>
          Value projection (<span style={{ color: CORAL }}>single shared</span>):
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <MatrixBlock label="X" dim="6×8" color="var(--text-muted)" width={96} height={72} />
          <Times color={SAGE} />
          <MatrixBlock label="W_V" dim="8×2" color={SAGE} width={24} height={96} />
          <Equals />
          <MatrixBlock label="V" dim="6×2" color={SAGE} width={24} height={72} />
          <span className="ml-2 rounded px-2 py-0.5 font-mono text-xs" style={{ backgroundColor: `${CORAL}20`, color: CORAL, border: `1px solid ${CORAL}40` }}>
            shared by all 4 heads
          </span>
        </div>
      </div>

      {/* Stats */}
      <div
        className="rounded p-3 mt-4"
        style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          <strong>KV cache per token:</strong> 1 K head + 1 V head = 2 vectors of size 2 = <span style={{ color: SAGE }}>4 values/token</span>
          <span style={{ color: CORAL }}> (4× smaller!)</span>
        </p>
        <p className="font-mono text-xs mt-1" style={{ color: "var(--text-muted)" }}>
          <strong>KV parameters:</strong> 1×(8×2) + 1×(8×2) = <span style={{ color: SAGE }}>32 parameters</span>
          <span style={{ color: CORAL }}> (4× fewer)</span>
        </p>
      </div>
    </div>
  );
}

function StepComputation() {
  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        Each query head computes attention against the <strong>same</strong> K and V.
        Different patterns still emerge because each head asks different &quot;questions&quot; (different Q projections).
      </p>

      {/* Show 4 heads all pointing to same K, V */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {HC.map((c, i) => (
          <div
            key={i}
            className="rounded border p-3"
            style={{ borderColor: c, backgroundColor: `${c}08` }}
          >
            <p className="mb-2 font-mono text-xs font-semibold" style={{ color: c }}>
              Head {i + 1}
            </p>
            <div className="flex items-center gap-1 flex-wrap">
              <MatrixBlock label={`Q${i + 1}`} dim="6×2" color={c} width={24} height={72} />
              <Times color={c} />
              <MatrixBlock label="K^T" dim="2×6" color={AMBER} width={72} height={24} />
              <Equals />
              <MatrixBlock label="scores" dim="6×6" color={c} width={72} height={72} />
            </div>
            <div className="mt-2 flex items-center gap-1 flex-wrap">
              <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
                softmax(scores/√d<sub>k</sub>)
              </span>
              <Times color={c} />
              <MatrixBlock label="V" dim="6×2" color={SAGE} width={24} height={72} />
              <Equals />
              <MatrixBlock label={`out${i + 1}`} dim="6×2" color={c} width={24} height={72} />
            </div>
          </div>
        ))}
      </div>

      <div
        className="rounded p-3 mt-4"
        style={{ borderLeft: `3px solid ${AMBER}`, backgroundColor: `${AMBER}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          Notice: the same <strong style={{ color: AMBER }}>K</strong> and{" "}
          <strong style={{ color: SAGE }}>V</strong> appear in every head.
          Only <strong style={{ color: CORAL }}>Q</strong> differs — that&apos;s where the diversity comes from.
          Each head &quot;asks different questions&quot; of the same information store.
        </p>
      </div>
    </div>
  );
}

function StepMemory() {
  const seqLens = [1024, 8192, 32768, 131072];
  const dk = 128;
  const heads = 32;
  const layers = 80;

  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        At real scale (Llama 3 70B: 32 KV heads, d<sub>k</sub>=128, 80 layers), the savings are dramatic.
        KV cache in GB (fp16) for a <strong>single sequence</strong>:
      </p>

      {/* Comparison table */}
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <th className="py-2 text-left" style={{ color: "var(--text-muted)" }}>Seq Length</th>
              <th className="py-2 text-right" style={{ color: "var(--text-muted)" }}>MHA (32 KV heads)</th>
              <th className="py-2 text-right" style={{ color: "var(--text-muted)" }}>MQA (1 KV head)</th>
              <th className="py-2 text-right" style={{ color: "var(--text-muted)" }}>Savings</th>
            </tr>
          </thead>
          <tbody>
            {seqLens.map((t) => {
              const mha = (2 * heads * t * dk * layers * 2) / 1e9;
              const mqa = (2 * 1 * t * dk * layers * 2) / 1e9;
              return (
                <tr key={t} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td className="py-2" style={{ color: "var(--text-secondary)" }}>{t.toLocaleString()}</td>
                  <td className="py-2 text-right" style={{ color: CORAL }}>{mha.toFixed(1)} GB</td>
                  <td className="py-2 text-right" style={{ color: SAGE }}>{mqa.toFixed(2)} GB</td>
                  <td className="py-2 text-right font-semibold" style={{ color: AMBER }}>{heads}×</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Visual bar comparison */}
      <div className="mt-5">
        <p className="mb-2 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          KV cache at 32K tokens (visual):
        </p>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-20 shrink-0 font-mono text-xs text-right" style={{ color: "var(--text-muted)" }}>MHA</span>
            <div className="flex-1 h-6 rounded overflow-hidden" style={{ backgroundColor: "var(--bg)" }}>
              <div className="h-full rounded" style={{ width: "100%", backgroundColor: CORAL, opacity: 0.6 }} />
            </div>
            <span className="w-16 shrink-0 font-mono text-xs" style={{ color: CORAL }}>83.9 GB</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 shrink-0 font-mono text-xs text-right" style={{ color: "var(--text-muted)" }}>MQA</span>
            <div className="flex-1 h-6 rounded overflow-hidden" style={{ backgroundColor: "var(--bg)" }}>
              <div className="h-full rounded" style={{ width: "3.1%", backgroundColor: SAGE, opacity: 0.6 }} />
            </div>
            <span className="w-16 shrink-0 font-mono text-xs" style={{ color: SAGE }}>2.6 GB</span>
          </div>
        </div>
      </div>

      <div
        className="rounded p-3 mt-4"
        style={{ borderLeft: `3px solid ${SAGE}`, backgroundColor: `${SAGE}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>The trade-off:</strong> all query heads now see identical keys and values.
          They can only differ in <em>what they ask for</em> (Q), not in <em>what information is available</em> (K, V).
          This limits pattern diversity — which is why GQA (a middle ground) became the industry standard.
        </p>
      </div>
    </div>
  );
}
