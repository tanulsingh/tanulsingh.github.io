"use client";

import { useState, useMemo } from "react";

const CORAL = "#E8976C";
const SAGE = "#8BAF7A";
const INDIGO = "#7B68EE";
const TEAL = "#45B7A0";
const AMBER = "#D4A843";
const MUTED = "#94a3b8";

const N = 6;
const B = 2;
const NUM_BLOCKS = N / B;

interface Step {
  j: number;
  i: number;
  desc: string;
  sramContents: string[];
  hbmWrites: string[];
  phase: "load_kv" | "compute" | "write_back";
}

function buildSteps(): Step[] {
  const steps: Step[] = [];
  for (let j = 0; j < NUM_BLOCKS; j++) {
    for (let i = 0; i < NUM_BLOCKS; i++) {
      steps.push({
        j,
        i,
        phase: "load_kv",
        desc: `Load K[${j}], V[${j}] and Q[${i}] with its running state (Õ[${i}], m[${i}], l[${i}]) from HBM into SRAM`,
        sramContents: [
          `Q[${i}]`,
          `K[${j}]`,
          `V[${j}]`,
          `Õ[${i}]`,
          `m[${i}]`,
          `l[${i}]`,
        ],
        hbmWrites: [],
      });
      steps.push({
        j,
        i,
        phase: "compute",
        desc: `Compute score tile S[${i},${j}] = Q[${i}] × K[${j}]ᵀ in SRAM. Apply online softmax correction. Accumulate into Õ[${i}]. Score tile is used and discarded — never touches HBM.`,
        sramContents: [
          `Q[${i}]`,
          `K[${j}]`,
          `V[${j}]`,
          `S[${i},${j}] ← computed`,
          `Õ[${i}] ← updated`,
          `m[${i}] ← updated`,
          `l[${i}] ← updated`,
        ],
        hbmWrites: [],
      });
      steps.push({
        j,
        i,
        phase: "write_back",
        desc: `Write updated Õ[${i}], m[${i}], l[${i}] back to HBM. Score tile S[${i},${j}] is discarded from SRAM forever.`,
        sramContents: [],
        hbmWrites: [`Õ[${i}]`, `m[${i}]`, `l[${i}]`],
      });
    }
  }
  return steps;
}

function AttentionMatrix({
  currentI,
  currentJ,
  phase,
  completedTiles,
}: {
  currentI: number;
  currentJ: number;
  phase: string;
  completedTiles: Set<string>;
}) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="font-mono text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
        Attention Score Matrix (N×N)
      </span>
      <div className="relative">
        {/* Column labels */}
        <div className="flex" style={{ marginLeft: 36 }}>
          {Array.from({ length: NUM_BLOCKS }).map((_, j) => (
            <div
              key={j}
              className="text-center font-mono text-xs"
              style={{
                width: NUM_BLOCKS * 32 / NUM_BLOCKS * B,
                color: j === currentJ ? INDIGO : MUTED,
                fontWeight: j === currentJ ? 700 : 400,
              }}
            >
              K[{j}]
            </div>
          ))}
        </div>
        <div className="flex">
          {/* Row labels */}
          <div className="flex flex-col justify-around" style={{ width: 36 }}>
            {Array.from({ length: NUM_BLOCKS }).map((_, i) => (
              <div
                key={i}
                className="font-mono text-xs text-right pr-1"
                style={{
                  height: 32 * B + 4,
                  lineHeight: `${32 * B + 4}px`,
                  color: i === currentI ? CORAL : MUTED,
                  fontWeight: i === currentI ? 700 : 400,
                }}
              >
                Q[{i}]
              </div>
            ))}
          </div>
          {/* Grid */}
          <div
            className="grid gap-1"
            style={{ gridTemplateColumns: `repeat(${NUM_BLOCKS}, 1fr)` }}
          >
            {Array.from({ length: NUM_BLOCKS * NUM_BLOCKS }).map((_, idx) => {
              const ti = Math.floor(idx / NUM_BLOCKS);
              const tj = idx % NUM_BLOCKS;
              const isCurrent = ti === currentI && tj === currentJ;
              const isCompleted = completedTiles.has(`${ti},${tj}`);
              const isActive = isCurrent && (phase === "compute");

              let bg = `${MUTED}12`;
              let border = `${MUTED}30`;
              let label = "";
              let labelColor = MUTED;

              if (isActive) {
                bg = `${TEAL}30`;
                border = TEAL;
                label = "IN SRAM";
                labelColor = TEAL;
              } else if (isCurrent && phase === "load_kv") {
                bg = `${AMBER}20`;
                border = `${AMBER}80`;
                label = "loading...";
                labelColor = AMBER;
              } else if (isCurrent && phase === "write_back") {
                bg = `${CORAL}15`;
                border = `${CORAL}50`;
                label = "discarded";
                labelColor = CORAL;
              } else if (isCompleted) {
                bg = `${SAGE}12`;
                border = `${SAGE}40`;
                label = "done";
                labelColor = `${SAGE}90`;
              }

              return (
                <div
                  key={idx}
                  className="rounded flex flex-col items-center justify-center"
                  style={{
                    width: 32 * B,
                    height: 32 * B,
                    backgroundColor: bg,
                    border: `2px solid ${border}`,
                    transition: "all 0.3s",
                  }}
                >
                  <span
                    className="font-mono text-xs"
                    style={{ color: isCurrent || isCompleted ? "var(--text-primary)" : `${MUTED}60` }}
                  >
                    S[{ti},{tj}]
                  </span>
                  {label && (
                    <span className="font-mono" style={{ fontSize: 9, color: labelColor }}>
                      {label}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <span className="text-xs mt-1" style={{ color: MUTED }}>
        Each tile is {B}×{B} scores. Only the active tile exists in SRAM — the rest are never stored.
      </span>
    </div>
  );
}

function MemoryPanel({
  label,
  color,
  items,
  subtitle,
}: {
  label: string;
  color: string;
  items: { name: string; highlight?: boolean; fading?: boolean }[];
  subtitle: string;
}) {
  return (
    <div
      className="rounded-lg border p-3 flex-1 min-w-[140px]"
      style={{ borderColor: `${color}50`, backgroundColor: `${color}06` }}
    >
      <div className="flex items-center gap-2 mb-1">
        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
        <span className="font-mono text-sm font-bold" style={{ color }}>
          {label}
        </span>
      </div>
      <div className="text-xs mb-2" style={{ color: MUTED }}>{subtitle}</div>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item, idx) => (
          <span
            key={idx}
            className="font-mono text-xs px-2 py-1 rounded-md"
            style={{
              backgroundColor: item.highlight
                ? `${TEAL}25`
                : item.fading
                  ? `${CORAL}15`
                  : `${color}15`,
              color: item.highlight ? TEAL : item.fading ? `${CORAL}80` : color,
              border: `1px solid ${item.highlight ? `${TEAL}40` : item.fading ? `${CORAL}30` : `${color}20`}`,
              textDecoration: item.fading ? "line-through" : "none",
            }}
          >
            {item.name}
          </span>
        ))}
        {items.length === 0 && (
          <span className="text-xs italic" style={{ color: `${MUTED}60` }}>empty</span>
        )}
      </div>
    </div>
  );
}

export function FlashAttentionVisualizer() {
  const [step, setStep] = useState(0);

  const steps = useMemo(() => buildSteps(), []);
  const current = steps[step];

  const completedTiles = useMemo(() => {
    const set = new Set<string>();
    for (let s = 0; s < step; s++) {
      if (steps[s].phase === "write_back") {
        set.add(`${steps[s].i},${steps[s].j}`);
      }
    }
    return set;
  }, [step, steps]);

  const hbmReads = useMemo(() => {
    let count = 0;
    for (let s = 0; s <= step; s++) {
      if (steps[s].phase === "load_kv") count++;
    }
    return count;
  }, [step, steps]);

  const hbmWrites = useMemo(() => {
    let count = 0;
    for (let s = 0; s <= step; s++) {
      if (steps[s].phase === "write_back") count++;
    }
    return count;
  }, [step, steps]);

  const sramItems = current.sramContents.map((name) => ({
    name,
    highlight: name.includes("S["),
    fading: false,
  }));

  const hbmItems = [
    { name: "Q (full)", fading: false },
    { name: "K (full)", fading: false },
    { name: "V (full)", fading: false },
    ...Array.from({ length: NUM_BLOCKS }).map((_, i) => ({
      name: `Õ[${i}], m[${i}], l[${i}]`,
      fading: false,
    })),
  ];

  if (current.phase === "write_back") {
    sramItems.push(
      { name: `S[${current.i},${current.j}]`, highlight: false, fading: true },
    );
  }

  return (
    <div
      className="my-8 rounded-xl border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-card)" }}
    >
      <div className="mb-1 text-base font-bold" style={{ color: "var(--text-primary)" }}>
        Flash Attention 1: Step by Step
      </div>
      <div className="mb-4 text-sm" style={{ color: MUTED }}>
        Watch how attention scores are computed tile-by-tile in SRAM without ever materializing
        the full N×N matrix in HBM. {N} tokens, block size {B} = {NUM_BLOCKS} blocks.
      </div>

      {/* Attention matrix */}
      <div className="flex justify-center mb-4">
        <AttentionMatrix
          currentI={current.i}
          currentJ={current.j}
          phase={current.phase}
          completedTiles={completedTiles}
        />
      </div>

      {/* Memory panels */}
      <div className="flex gap-3 mb-4 flex-wrap">
        <MemoryPanel
          label="SRAM"
          color={TEAL}
          items={sramItems}
          subtitle="Fast, tiny (~20 MB) — the workspace"
        />
        <MemoryPanel
          label="HBM"
          color={CORAL}
          items={hbmItems}
          subtitle="Slow, large (80 GB) — the storage"
        />
      </div>

      {/* Step description */}
      <div
        className="rounded-lg border p-3 mb-4 text-sm"
        style={{
          borderColor: "var(--border)",
          backgroundColor: "var(--bg-elevated)",
          color: "var(--text-primary)",
        }}
      >
        <span className="font-mono text-xs mr-2" style={{ color: MUTED }}>
          Step {step + 1}/{steps.length}:
        </span>
        {current.desc}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setStep(0)}
            disabled={step === 0}
            className="px-2 py-1 rounded text-xs font-mono"
            style={{
              color: step === 0 ? `${MUTED}40` : MUTED,
              border: `1px solid ${MUTED}30`,
            }}
          >
            Reset
          </button>
          <button
            onClick={() => setStep(Math.max(0, step - 1))}
            disabled={step === 0}
            className="px-2 py-1 rounded text-xs font-mono"
            style={{
              color: step === 0 ? `${MUTED}40` : MUTED,
              border: `1px solid ${MUTED}30`,
            }}
          >
            ⟨ Prev
          </button>
          <button
            onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
            disabled={step === steps.length - 1}
            className="px-3 py-1 rounded text-xs font-mono font-semibold"
            style={{
              color: step === steps.length - 1 ? `${MUTED}40` : TEAL,
              border: `1px solid ${step === steps.length - 1 ? `${MUTED}30` : `${TEAL}60`}`,
            }}
          >
            Next ⟩
          </button>
        </div>
        <div className="flex gap-4 text-xs font-mono">
          <span style={{ color: MUTED }}>
            HBM reads: <span style={{ color: CORAL }}>{hbmReads}</span>
          </span>
          <span style={{ color: MUTED }}>
            HBM writes: <span style={{ color: CORAL }}>{hbmWrites}</span>
          </span>
          <span style={{ color: MUTED }}>
            Score tiles in HBM: <span style={{ color: SAGE }}>0 (always)</span>
          </span>
        </div>
      </div>
    </div>
  );
}
