"use client";

import { useState } from "react";

const CORAL = "#E8976C";
const AMBER = "#D4A843";
const SAGE = "#8BAF7A";
const INDIGO = "#7B68EE";
const TEAL = "#45B7A0";
const HC = ["#5B9BD5", "#9B59B6", "#E85B7A", "#45B7A0"];

function MatrixBlock({
  label,
  dim,
  color,
  cols = 2,
  rows = 6,
  highlight = false,
}: {
  label: string;
  dim: string;
  color: string;
  cols?: number;
  rows?: number;
  highlight?: boolean;
}) {
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
          backgroundColor: highlight ? `${color}15` : `${color}08`,
          boxShadow: highlight ? `0 0 8px ${color}40` : "none",
        }}
      >
        <div className="grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: "1px" }}>
          {Array.from({ length: rows * cols }).map((_, idx) => (
            <div
              key={idx}
              className="rounded-sm"
              style={{ width: "10px", height: "10px", backgroundColor: color, opacity: highlight ? 0.5 : 0.25 }}
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

function Arrow({ color = "var(--text-muted)", label }: { color?: string; label?: string }) {
  return (
    <div className="flex flex-col items-center px-1">
      {label && (
        <span className="font-mono" style={{ fontSize: "8px", color, marginBottom: "2px" }}>
          {label}
        </span>
      )}
      <svg width="28" height="12" viewBox="0 0 28 12">
        <path d="M0 6 L22 6 M18 2 L24 6 L18 10" fill="none" stroke={color} strokeWidth="1.5" />
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
    title: "The sharing problem",
    subtitle: "Why MQA/GQA hit a wall",
  },
  {
    title: "The compression insight",
    subtitle: "High-dimensional KV vectors are mostly redundant",
  },
  {
    title: "The MLA mechanism",
    subtitle: "Compress → cache → decompress per head",
  },
  {
    title: "The absorption trick",
    subtitle: "Why decompression is actually free",
  },
  {
    title: "The numbers",
    subtitle: "4× better than GQA at DeepSeek scale",
  },
];

export function MLAVisualizer() {
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
        Multi-Latent Attention — From Intuition to Mechanism
      </h4>

      {/* Step navigation */}
      <div className="mb-5 flex items-center gap-2 flex-wrap">
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
      <div className="min-h-[360px]">
        {step === 0 && <StepProblem />}
        {step === 1 && <StepInsight />}
        {step === 2 && <StepMechanism />}
        {step === 3 && <StepAbsorption />}
        {step === 4 && <StepNumbers />}
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

function StepProblem() {
  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        MQA and GQA save memory by <strong>sharing</strong> — multiple query heads look at the same K, V.
        But sharing has a fundamental limit: shared heads see identical information.
      </p>

      {/* Visual: sharing forces identical views */}
      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded border p-3" style={{ borderColor: "var(--border)" }}>
          <p className="mb-2 font-mono text-xs font-semibold" style={{ color: SAGE }}>
            GQA (4 heads, 2 groups)
          </p>
          <div className="flex items-center gap-2">
            <div className="flex flex-col gap-1">
              <div className="flex gap-1">
                <MatrixBlock label="Q₁" dim="" color={HC[0]} cols={2} rows={4} />
                <MatrixBlock label="Q₂" dim="" color={HC[1]} cols={2} rows={4} />
              </div>
              <span className="text-center font-mono" style={{ fontSize: "8px", color: "var(--text-muted)" }}>
                share K₁, V₁
              </span>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex gap-1">
                <MatrixBlock label="Q₃" dim="" color={HC[2]} cols={2} rows={4} />
                <MatrixBlock label="Q₄" dim="" color={HC[3]} cols={2} rows={4} />
              </div>
              <span className="text-center font-mono" style={{ fontSize: "8px", color: "var(--text-muted)" }}>
                share K₂, V₂
              </span>
            </div>
          </div>
          <p className="mt-2 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
            Q₁ and Q₂ must attend to the same keys/values
          </p>
        </div>

        <div className="rounded border p-3" style={{ borderColor: CORAL, backgroundColor: `${CORAL}08` }}>
          <p className="mb-2 font-mono text-xs font-semibold" style={{ color: CORAL }}>
            The dilemma
          </p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>More sharing</span>
              <Arrow color={CORAL} />
              <span className="font-mono text-xs" style={{ color: CORAL }}>Less memory</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>More sharing</span>
              <Arrow color={CORAL} />
              <span className="font-mono text-xs" style={{ color: CORAL }}>Less diversity</span>
            </div>
            <div
              className="mt-3 rounded p-2"
              style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
            >
              <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-secondary)" }}>
                Can we reduce memory <em>without</em> forcing heads to see identical information?
              </p>
            </div>
          </div>
        </div>
      </div>

      <div
        className="rounded p-3"
        style={{ borderLeft: `3px solid ${INDIGO}`, backgroundColor: `${INDIGO}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>The question MLA asks:</strong> What if instead of storing fewer KV sets (sharing),
          we store <em>smaller</em> KV representations (compression)? Each head gets unique information,
          but we store it in a compact form.
        </p>
      </div>
    </div>
  );
}

function StepInsight() {
  const fullDim = 16;
  const usedDim = 4;

  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        A token&apos;s K vector in a large model might be 5120-dimensional. But how much of that space
        actually carries unique information?
      </p>

      {/* Visual: full-dimensional vector with mostly redundant dims */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: AMBER }}>
          A typical K vector (d_model = 5120):
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex gap-px">
            {Array.from({ length: fullDim }).map((_, i) => (
              <div
                key={i}
                className="rounded-sm"
                style={{
                  width: "18px",
                  height: "36px",
                  backgroundColor: i < usedDim ? INDIGO : "var(--text-muted)",
                  opacity: i < usedDim ? 0.7 : 0.12,
                  border: `1px solid ${i < usedDim ? INDIGO : "var(--border)"}`,
                }}
              />
            ))}
          </div>
        </div>
        <div className="mt-1 flex gap-4">
          <span className="flex items-center gap-1 font-mono" style={{ fontSize: "9px" }}>
            <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: INDIGO, opacity: 0.7 }} />
            <span style={{ color: INDIGO }}>Informative dimensions</span>
          </span>
          <span className="flex items-center gap-1 font-mono" style={{ fontSize: "9px" }}>
            <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: "var(--text-muted)", opacity: 0.12, border: "1px solid var(--border)" }} />
            <span style={{ color: "var(--text-muted)" }}>Redundant / correlated</span>
          </span>
        </div>
      </div>

      {/* Analogy */}
      <div className="mb-4 rounded border p-3" style={{ borderColor: "var(--border)" }}>
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: INDIGO }}>
          The analogy: JPEG for attention
        </p>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          JPEG doesn&apos;t store every pixel. It stores a compressed representation and reconstructs the image.
          Most of the &quot;information&quot; in a high-res image is redundant — smooth gradients, repeated patterns.
          Same principle: most of the 5120 dimensions in a K vector are correlated and compressible.
        </p>
      </div>

      {/* Compression visual */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: SAGE }}>
          The MLA idea: compress into a latent space
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex flex-col items-center">
            <span className="font-mono" style={{ fontSize: "9px", color: AMBER }}>full KV</span>
            <div className="flex gap-px">
              {Array.from({ length: fullDim }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-sm"
                  style={{ width: "12px", height: "28px", backgroundColor: AMBER, opacity: 0.3 }}
                />
              ))}
            </div>
            <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>5120 dims</span>
          </div>
          <Arrow color={INDIGO} label="compress" />
          <div className="flex flex-col items-center">
            <span className="font-mono" style={{ fontSize: "9px", color: INDIGO }}>latent c_t</span>
            <div className="flex gap-px">
              {Array.from({ length: usedDim }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-sm"
                  style={{ width: "12px", height: "28px", backgroundColor: INDIGO, opacity: 0.6 }}
                />
              ))}
            </div>
            <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>512 dims</span>
          </div>
          <Arrow color={TEAL} label="decompress" />
          <div className="flex flex-col items-center">
            <span className="font-mono" style={{ fontSize: "9px", color: TEAL }}>reconstructed K</span>
            <div className="flex gap-px">
              {Array.from({ length: fullDim }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-sm"
                  style={{ width: "12px", height: "28px", backgroundColor: TEAL, opacity: 0.3 }}
                />
              ))}
            </div>
            <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>5120 dims</span>
          </div>
        </div>
      </div>

      <div
        className="rounded p-3"
        style={{ borderLeft: `3px solid ${SAGE}`, backgroundColor: `${SAGE}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>Key difference from sharing:</strong> Each head can have its own decompression matrix.
          From the same 512-dim latent, head 1 reconstructs <em>different</em> keys than head 2.
          Full per-head diversity, 10× less storage.
        </p>
      </div>
    </div>
  );
}

function StepMechanism() {
  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        For each token x<sub>t</sub>, MLA performs: compress once, cache the latent, decompress per-head during attention.
      </p>

      {/* Step 1: Compression */}
      <div className="mb-4 rounded border p-3" style={{ borderColor: INDIGO, backgroundColor: `${INDIGO}06` }}>
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: INDIGO }}>
          Step 1: Compress (happens once per token)
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <MatrixBlock label="x_t" dim="1×5120" color="var(--text-muted)" cols={10} rows={1} />
          <Times color={INDIGO} />
          <MatrixBlock label="W_down" dim="5120×512" color={INDIGO} cols={2} rows={10} />
          <Equals />
          <MatrixBlock label="c_t" dim="1×512" color={INDIGO} cols={4} rows={1} highlight />
        </div>
        <p className="mt-2 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
          This c_t is what gets cached. 10× smaller than caching full K and V.
        </p>
      </div>

      {/* Step 2: Cache */}
      <div className="mb-4 rounded border p-3" style={{ borderColor: SAGE, backgroundColor: `${SAGE}06` }}>
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: SAGE }}>
          Step 2: Cache (the latent, not K/V)
        </p>
        <div className="flex items-center gap-1">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="flex flex-col items-center">
              <span className="font-mono" style={{ fontSize: "8px", color: "var(--text-muted)" }}>
                t{i + 1}
              </span>
              <div
                className="rounded"
                style={{
                  width: "28px",
                  height: "16px",
                  backgroundColor: INDIGO,
                  opacity: 0.4,
                  border: `1px solid ${INDIGO}`,
                }}
              />
            </div>
          ))}
          <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>...</span>
        </div>
        <p className="mt-2 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
          KV cache stores [seq_len × 512] instead of [seq_len × 5120 × 2]
        </p>
      </div>

      {/* Step 3: Decompress per head */}
      <div className="mb-4 rounded border p-3" style={{ borderColor: CORAL, backgroundColor: `${CORAL}06` }}>
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: CORAL }}>
          Step 3: Decompress (per head, during attention)
        </p>
        <div className="space-y-2">
          {HC.slice(0, 3).map((c, i) => (
            <div key={i} className="flex items-center gap-2 flex-wrap">
              <MatrixBlock label="c_t" dim="" color={INDIGO} cols={4} rows={1} />
              <Times color={c} />
              <MatrixBlock label={`W_up_K${i + 1}`} dim="" color={c} cols={5} rows={2} />
              <Equals />
              <MatrixBlock label={`K${i + 1}`} dim="" color={c} cols={5} rows={1} />
              <span className="font-mono" style={{ fontSize: "8px", color: c }}>← unique to head {i + 1}</span>
            </div>
          ))}
          <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            ... same for V with separate W_up_V per head
          </span>
        </div>
      </div>

      <div
        className="rounded p-3"
        style={{ borderLeft: `3px solid ${TEAL}`, backgroundColor: `${TEAL}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>Each head reconstructs its own unique K, V</strong> from the shared latent.
          Unlike GQA where heads literally see the same K/V, here each head has a different
          &quot;lens&quot; (W_up) into the compressed information.
        </p>
      </div>
    </div>
  );
}

function StepAbsorption() {
  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        Naive MLA would decompress K for every cached token on every step — expensive.
        The &quot;absorption trick&quot; eliminates this entirely.
      </p>

      {/* Naive approach */}
      <div className="mb-4 rounded border p-3" style={{ borderColor: CORAL, backgroundColor: `${CORAL}06` }}>
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: CORAL }}>
          Naive (expensive):
        </p>
        <div className="space-y-1">
          <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
            score = q · k<sup>T</sup>
          </p>
          <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
            &nbsp;&nbsp;&nbsp;&nbsp; = q · (c · W_up<sup>K</sup>)<sup>T</sup>
          </p>
          <p className="font-mono text-xs" style={{ color: CORAL }}>
            ↑ must decompress c for every cached token, every step
          </p>
        </div>
      </div>

      {/* Absorbed approach */}
      <div className="mb-4 rounded border p-3" style={{ borderColor: SAGE, backgroundColor: `${SAGE}06` }}>
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: SAGE }}>
          Absorbed (free):
        </p>
        <div className="space-y-1">
          <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
            score = q · (c · W_up<sup>K</sup>)<sup>T</sup>
          </p>
          <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
            &nbsp;&nbsp;&nbsp;&nbsp; = q · W_up<sup>K<sup>T</sup></sup> · c<sup>T</sup>
          </p>
          <p className="font-mono text-xs" style={{ color: SAGE }}>
            &nbsp;&nbsp;&nbsp;&nbsp; = <strong>q&apos;</strong> · c<sup>T</sup>
            &nbsp;&nbsp;where q&apos; = q · W_up<sup>K<sup>T</sup></sup>
          </p>
        </div>
      </div>

      {/* Visual comparison */}
      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded border p-3" style={{ borderColor: CORAL }}>
          <p className="mb-2 font-mono text-xs" style={{ color: CORAL }}>Naive: ops per attention step</p>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
                Decompress all cached c → K
              </span>
              <span className="font-mono text-xs font-semibold" style={{ color: CORAL }}>O(n·d)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
                Compute q · K<sup>T</sup>
              </span>
              <span className="font-mono text-xs font-semibold" style={{ color: CORAL }}>O(n·d)</span>
            </div>
          </div>
        </div>
        <div className="rounded border p-3" style={{ borderColor: SAGE }}>
          <p className="mb-2 font-mono text-xs" style={{ color: SAGE }}>Absorbed: ops per attention step</p>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
                Transform q → q&apos;
              </span>
              <span className="font-mono text-xs font-semibold" style={{ color: SAGE }}>O(d²) once</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
                Compute q&apos; · c<sup>T</sup>
              </span>
              <span className="font-mono text-xs font-semibold" style={{ color: SAGE }}>O(n·d_c)</span>
            </div>
          </div>
        </div>
      </div>

      <div
        className="rounded p-3"
        style={{ borderLeft: `3px solid ${INDIGO}`, backgroundColor: `${INDIGO}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>The trick:</strong> transform the query once (cheap — it&apos;s a single vector),
          then dot it directly against the compressed latents. The decompression matrix
          &quot;absorbs&quot; into the query. Full K is never materialized. Same trick applies to the value side.
        </p>
      </div>
    </div>
  );
}

function StepNumbers() {
  const methods = [
    { name: "MHA (32 heads)", cache: 2 * 32 * 128, color: CORAL, pct: 100 },
    { name: "GQA (8 groups)", cache: 2 * 8 * 128, color: AMBER, pct: 25 },
    { name: "MQA (1 head)", cache: 2 * 1 * 128, color: SAGE, pct: 3.1 },
    { name: "MLA (d_c=512)", cache: 512, color: INDIGO, pct: 6.25 },
  ];

  return (
    <div>
      <p className="mb-4 text-sm" style={{ color: "var(--text-secondary)" }}>
        DeepSeek-V2: d<sub>model</sub>=5120, d<sub>c</sub>=512, 32 heads, d<sub>k</sub>=128.
        KV cache per token (values stored):
      </p>

      {/* Bar chart comparison */}
      <div className="mb-4 space-y-2">
        {methods.map((m) => (
          <div key={m.name} className="flex items-center gap-2">
            <span className="w-28 shrink-0 text-right font-mono text-xs" style={{ color: "var(--text-muted)" }}>
              {m.name}
            </span>
            <div className="flex-1 h-5 rounded overflow-hidden" style={{ backgroundColor: "var(--bg)" }}>
              <div
                className="h-full rounded transition-all"
                style={{ width: `${m.pct}%`, backgroundColor: m.color, opacity: 0.6, minWidth: m.pct < 5 ? "4px" : undefined }}
              />
            </div>
            <span className="w-24 shrink-0 font-mono text-xs" style={{ color: m.color }}>
              {m.cache.toLocaleString()} vals
            </span>
          </div>
        ))}
      </div>

      {/* Comparison table */}
      <div className="mb-4 overflow-x-auto">
        <table className="w-full font-mono text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <th className="py-2 text-left" style={{ color: "var(--text-muted)" }}>Method</th>
              <th className="py-2 text-right" style={{ color: "var(--text-muted)" }}>Cache/token</th>
              <th className="py-2 text-right" style={{ color: "var(--text-muted)" }}>vs MHA</th>
              <th className="py-2 text-right" style={{ color: "var(--text-muted)" }}>Per-head diversity?</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td className="py-2" style={{ color: CORAL }}>MHA</td>
              <td className="py-2 text-right">8,192</td>
              <td className="py-2 text-right">1×</td>
              <td className="py-2 text-right" style={{ color: SAGE }}>Yes</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td className="py-2" style={{ color: AMBER }}>GQA-8</td>
              <td className="py-2 text-right">2,048</td>
              <td className="py-2 text-right">4×</td>
              <td className="py-2 text-right" style={{ color: CORAL }}>Partial (grouped)</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td className="py-2" style={{ color: SAGE }}>MQA</td>
              <td className="py-2 text-right">256</td>
              <td className="py-2 text-right">32×</td>
              <td className="py-2 text-right" style={{ color: CORAL }}>No (all shared)</td>
            </tr>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              <td className="py-2 font-semibold" style={{ color: INDIGO }}>MLA</td>
              <td className="py-2 text-right font-semibold">512</td>
              <td className="py-2 text-right font-semibold">16×</td>
              <td className="py-2 text-right font-semibold" style={{ color: SAGE }}>Yes (via W_up)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div
        className="rounded p-3"
        style={{ borderLeft: `3px solid ${INDIGO}`, backgroundColor: `${INDIGO}08` }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong>MLA&apos;s unique position:</strong> 16× compression (between GQA and MQA in cache size),
          but with full per-head diversity (like MHA). It&apos;s the only method that achieves both.
          DeepSeek-V2 matches Llama 2 70B quality with 21B active parameters and this cache design.
          DeepSeek-V3 (671B) validated it at frontier scale.
        </p>
      </div>
    </div>
  );
}
