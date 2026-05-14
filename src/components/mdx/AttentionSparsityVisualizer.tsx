"use client";

import { useState, useMemo } from "react";

const CORAL = "#E8976C";
const SAGE = "#8BAF7A";
const MUTED = "#94a3b8";

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function generateRealisticAttention(seqLen: number, seed: number): number[][] {
  const rand = seededRandom(seed);
  const weights: number[][] = [];

  for (let i = 0; i < seqLen; i++) {
    const scores: number[] = [];
    for (let j = 0; j <= i; j++) {
      let score = -2 + rand() * 1.5;

      // Strong local bias: nearby tokens get much higher scores
      const distance = i - j;
      if (distance <= 3) score += 3.5 + rand() * 1.5;
      else if (distance <= 6) score += 1.5 + rand() * 0.8;

      // A few random "important" distant tokens (simulating coreference, etc.)
      if (distance > 5 && rand() < 0.06) score += 3.0 + rand() * 1.0;

      // First token often gets attention (BOS/positional bias)
      if (j === 0) score += 1.2 + rand() * 0.5;

      scores.push(score);
    }
    // Pad future positions with -inf (causal)
    for (let j = i + 1; j < seqLen; j++) {
      scores.push(-1e9);
    }
    weights.push(softmax(scores));
  }
  return weights;
}

function HeatmapView({
  weights,
  tokens,
  selectedRow,
  onSelectRow,
  threshold,
}: {
  weights: number[][];
  tokens: string[];
  selectedRow: number | null;
  onSelectRow: (i: number) => void;
  threshold: number;
}) {
  const n = tokens.length;
  const cellSize = 36;

  return (
    <div className="overflow-x-auto">
      <div className="inline-block">
        {/* Column headers */}
        <div className="flex" style={{ marginLeft: 52 }}>
          {tokens.map((t, j) => (
            <div
              key={j}
              className="text-center font-mono"
              style={{
                width: cellSize,
                fontSize: 9,
                color: MUTED,
                whiteSpace: "nowrap",
              }}
            >
              {t}
            </div>
          ))}
        </div>
        {/* Rows */}
        {weights.map((row, i) => (
          <div
            key={i}
            className="flex items-center cursor-pointer"
            onClick={() => onSelectRow(i)}
            style={{
              opacity: selectedRow !== null && selectedRow !== i ? 0.4 : 1,
              transition: "opacity 0.2s",
            }}
          >
            <span
              className="font-mono text-right pr-2"
              style={{
                width: 52,
                fontSize: 9,
                color: selectedRow === i ? CORAL : MUTED,
                fontWeight: selectedRow === i ? 600 : 400,
                whiteSpace: "nowrap",
              }}
            >
              {tokens[i]}
            </span>
            {row.map((w, j) => {
              const isAboveThreshold = w >= threshold;
              const isMasked = j > i;
              return (
                <div
                  key={j}
                  style={{
                    width: cellSize - 2,
                    height: cellSize - 2,
                    margin: "1px",
                    borderRadius: 3,
                    backgroundColor: isMasked
                      ? "transparent"
                      : isAboveThreshold
                        ? `rgba(232, 151, 108, ${Math.min(w * 4, 1)})`
                        : `rgba(148, 163, 184, ${w * 3})`,
                    border:
                      selectedRow === i && isAboveThreshold && !isMasked
                        ? `1px solid ${CORAL}`
                        : "1px solid transparent",
                  }}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function DistributionBar({
  weights,
  selectedRow,
  threshold,
}: {
  weights: number[][];
  selectedRow: number | null;
  threshold: number;
}) {
  const row = selectedRow !== null ? weights[selectedRow] : null;
  if (!row) return null;

  const causalRow = row.filter((_, j) => j <= (selectedRow ?? 0));
  const sorted = [...causalRow].sort((a, b) => b - a);
  const total = sorted.reduce((a, b) => a + b, 0);
  const aboveThreshold = sorted.filter((w) => w >= threshold);
  const aboveSum = aboveThreshold.reduce((a, b) => a + b, 0);

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          Weight distribution for &ldquo;{/* tokens label handled by parent */}&rdquo;
        </span>
      </div>
      {/* Sorted bar chart */}
      <div className="flex items-end gap-px" style={{ height: 60 }}>
        {sorted.map((w, idx) => (
          <div
            key={idx}
            style={{
              width: Math.max(2, 200 / sorted.length),
              height: `${(w / sorted[0]) * 100}%`,
              backgroundColor: w >= threshold ? CORAL : `${MUTED}40`,
              borderRadius: "1px 1px 0 0",
              minHeight: 1,
            }}
          />
        ))}
      </div>
      {/* Stats */}
      <div className="mt-2 flex gap-4 text-xs" style={{ color: MUTED }}>
        <span>
          <span style={{ color: CORAL, fontWeight: 600 }}>{aboveThreshold.length}</span> of{" "}
          {causalRow.length} tokens carry{" "}
          <span style={{ color: CORAL, fontWeight: 600 }}>{(aboveSum / total * 100).toFixed(0)}%</span>{" "}
          of attention weight
        </span>
        <span>
          <span style={{ color: MUTED }}>{causalRow.length - aboveThreshold.length}</span> tokens share the remaining{" "}
          {((1 - aboveSum / total) * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function WasteCalculator({
  weights,
  threshold,
  seqLen,
}: {
  weights: number[][];
  threshold: number;
  seqLen: number;
}) {
  const stats = useMemo(() => {
    let totalScores = 0;
    let meaningfulScores = 0;
    let totalWeight = 0;
    let meaningfulWeight = 0;

    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j <= i; j++) {
        totalScores++;
        totalWeight += weights[i][j];
        if (weights[i][j] >= threshold) {
          meaningfulScores++;
          meaningfulWeight += weights[i][j];
        }
      }
    }

    return {
      totalScores,
      meaningfulScores,
      wastedPct: ((1 - meaningfulScores / totalScores) * 100).toFixed(1),
      weightCaptured: ((meaningfulWeight / totalWeight) * 100).toFixed(1),
    };
  }, [weights, threshold]);

  return (
    <div
      className="mt-4 rounded-lg border p-4"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-elevated)" }}
    >
      <div className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
        The Waste at Scale
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <div style={{ color: MUTED }}>Total attention scores computed</div>
          <div className="font-mono font-semibold" style={{ color: "var(--text-primary)" }}>
            {stats.totalScores.toLocaleString()}
          </div>
        </div>
        <div>
          <div style={{ color: MUTED }}>Scores above threshold ({(threshold * 100).toFixed(0)}%)</div>
          <div className="font-mono font-semibold" style={{ color: SAGE }}>
            {stats.meaningfulScores.toLocaleString()}
          </div>
        </div>
        <div>
          <div style={{ color: MUTED }}>Wasted compute</div>
          <div className="font-mono font-semibold" style={{ color: CORAL }}>
            {stats.wastedPct}%
          </div>
        </div>
        <div>
          <div style={{ color: MUTED }}>Weight captured by meaningful scores</div>
          <div className="font-mono font-semibold" style={{ color: SAGE }}>
            {stats.weightCaptured}%
          </div>
        </div>
      </div>
      <div className="mt-3 text-xs" style={{ color: MUTED }}>
        At this {seqLen}-token scale, {stats.wastedPct}% of attention scores contribute almost nothing.
        Now imagine a 128K-token sequence — that&apos;s billions of near-zero scores computed and discarded every forward pass.
      </div>
    </div>
  );
}

export function AttentionSparsityVisualizer() {
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(0.05);

  const tokens = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired", "and", "cold"];
  const seqLen = tokens.length;

  const weights = useMemo(() => generateRealisticAttention(seqLen, 42), [seqLen]);

  return (
    <div
      className="my-8 rounded-xl border p-5"
      style={{
        borderColor: "var(--border)",
        backgroundColor: "var(--bg-card)",
      }}
    >
      <div className="mb-1 text-base font-bold" style={{ color: "var(--text-primary)" }}>
        Attention Weight Sparsity
      </div>
      <div className="mb-4 text-sm" style={{ color: MUTED }}>
        A simulated causal attention heatmap for a 16-token sentence. Click any row to see how that
        token distributes its attention. Bright cells = meaningful weight. Most cells are near-zero.
      </div>

      {/* Threshold control */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-xs" style={{ color: MUTED }}>
          &ldquo;Meaningful&rdquo; threshold:
        </span>
        <input
          type="range"
          min={1}
          max={15}
          value={threshold * 100}
          onChange={(e) => setThreshold(Number(e.target.value) / 100)}
          className="w-32"
          style={{ accentColor: CORAL }}
        />
        <span className="font-mono text-xs" style={{ color: CORAL }}>
          {(threshold * 100).toFixed(0)}%
        </span>
      </div>

      {/* Heatmap */}
      <HeatmapView
        weights={weights}
        tokens={tokens}
        selectedRow={selectedRow}
        onSelectRow={setSelectedRow}
        threshold={threshold}
      />

      {/* Distribution for selected row */}
      {selectedRow !== null && (
        <DistributionBar weights={weights} selectedRow={selectedRow} threshold={threshold} />
      )}

      {/* Waste stats */}
      <WasteCalculator weights={weights} threshold={threshold} seqLen={seqLen} />
    </div>
  );
}
