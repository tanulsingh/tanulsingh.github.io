"use client";

import { useState, useMemo } from "react";

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// Seeded pseudo-random for consistent results
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function computeAttention(tokens: string[], causal: boolean): { weights: number[][]; qkScores: number[][] } {
  const n = tokens.length;
  const dk = 4;

  // Generate embeddings and projections using seeded random based on token content
  const rand = seededRandom(tokens.join("").split("").reduce((a, c) => a + c.charCodeAt(0), 0));

  const Q: number[][] = [];
  const K: number[][] = [];

  for (let i = 0; i < n; i++) {
    const q: number[] = [];
    const k: number[] = [];
    for (let d = 0; d < dk; d++) {
      q.push(rand() * 2 - 1);
      k.push(rand() * 2 - 1);
    }
    Q.push(q);
    K.push(k);
  }

  // Compute QK^T scores
  const scores: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      let score = 0;
      for (let d = 0; d < dk; d++) {
        score += Q[i][d] * K[j][d];
      }
      // Scale by sqrt(dk)
      score = score / Math.sqrt(dk);

      // Apply causal mask if enabled
      if (causal && j > i) score = -1e9;

      row.push(score);
    }
    scores.push(row);
  }

  // Apply softmax per row
  const weights = scores.map((row) => softmax(row));

  return { weights, qkScores: scores };
}

export function AttentionVisualizer() {
  const [input, setInput] = useState("The cat sat on the mat because it was tired");
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [causal, setCausal] = useState(false);

  const tokens = useMemo(() => input.split(/\s+/).filter(Boolean), [input]);
  const { weights } = useMemo(() => {
    if (tokens.length === 0) return { weights: [], qkScores: [] };
    return computeAttention(tokens, causal);
  }, [tokens, causal]);

  const maxWeight = useMemo(() => {
    let max = 0;
    for (const row of weights) {
      for (const w of row) {
        if (w > max) max = w;
      }
    }
    return max;
  }, [weights]);

  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        interactive
      </p>
      <h4 className="mb-4 text-lg font-bold" style={{ fontFamily: "var(--font-serif)" }}>
        Self-Attention — Visualized
      </h4>

      {/* Input */}
      <div className="mb-4">
        <label className="mb-1 block font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          input sentence:
        </label>
        <input
          type="text"
          value={input}
          onChange={(e) => { setInput(e.target.value); setSelectedToken(null); }}
          className="w-full rounded border px-3 py-2 font-mono text-sm outline-none"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--bg)", color: "var(--text)" }}
        />
        <p className="mt-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          click any token below to see what it attends to
        </p>
      </div>

      {/* Mode toggle */}
      <div className="mb-4 flex items-center gap-3">
        <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>mode:</span>
        <button
          onClick={() => { setCausal(false); setSelectedToken(null); }}
          className="rounded px-2.5 py-1 font-mono text-xs"
          style={{
            backgroundColor: !causal ? "var(--tag-bg)" : "transparent",
            color: !causal ? "var(--primary)" : "var(--text-muted)",
            border: "1px solid var(--border)",
          }}
        >
          bidirectional (encoder)
        </button>
        <button
          onClick={() => { setCausal(true); setSelectedToken(null); }}
          className="rounded px-2.5 py-1 font-mono text-xs"
          style={{
            backgroundColor: causal ? "var(--tag-bg)" : "transparent",
            color: causal ? "var(--primary)" : "var(--text-muted)",
            border: "1px solid var(--border)",
          }}
        >
          causal (decoder / GPT)
        </button>
      </div>

      {/* Token flow visualization */}
      {tokens.length > 0 && tokens.length <= 14 && (
        <div className="mb-6">
          <p className="mb-3 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            {selectedToken !== null
              ? `"${tokens[selectedToken]}" (pos ${selectedToken}) attends to:`
              : "click a token to see its attention flow"}
          </p>

          {/* SVG arrow flow diagram */}
          <div className="relative">
            <svg
              viewBox={`0 0 ${Math.max(tokens.length * 80, 400)} 180`}
              className="w-full"
              style={{ minHeight: "180px" }}
            >
              {/* Token boxes at the bottom */}
              {tokens.map((token, i) => {
                const x = 40 + i * 70;
                const y = 140;
                const isSelected = selectedToken === i;
                const attendedWeight = selectedToken !== null ? weights[selectedToken]?.[i] || 0 : 0;
                const isAttended = selectedToken !== null && attendedWeight > 0.05 && i !== selectedToken;

                return (
                  <g key={i} onClick={() => setSelectedToken(selectedToken === i ? null : i)} className="cursor-pointer">
                    {/* Token box */}
                    <rect
                      x={x - 28} y={y - 12} width={56} height={24} rx={4}
                      fill={isSelected ? "var(--primary)" : isAttended ? `rgba(232,151,108,${0.15 + attendedWeight * 0.5})` : "var(--bg)"}
                      stroke={isSelected ? "var(--primary)" : isAttended ? "var(--primary)" : "var(--border)"}
                      strokeWidth={isSelected || isAttended ? 1.5 : 1}
                    />
                    {/* Token text */}
                    <text
                      x={x} y={y + 3}
                      textAnchor="middle" dominantBaseline="middle"
                      fill={isSelected ? "white" : "var(--text-secondary)"}
                      fontSize="10" fontFamily="var(--font-mono)"
                    >
                      {token.length > 6 ? token.slice(0, 5) + "…" : token}
                    </text>
                    {/* Weight label */}
                    {isAttended && (
                      <text
                        x={x} y={y + 22}
                        textAnchor="middle"
                        fill="var(--primary)"
                        fontSize="8" fontFamily="var(--font-mono)"
                      >
                        {(attendedWeight * 100).toFixed(0)}%
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Curved arrows from selected token to attended tokens */}
              {selectedToken !== null && tokens.map((_, j) => {
                if (j === selectedToken) return null;
                if (causal && j > selectedToken) return null;
                const weight = weights[selectedToken]?.[j] || 0;
                if (weight < 0.03) return null;

                const fromX = 40 + selectedToken * 70;
                const toX = 40 + j * 70;
                const fromY = 128;
                const toY = 128;

                // Arc height proportional to distance
                const dist = Math.abs(selectedToken - j);
                const arcHeight = 30 + dist * 15;
                const midX = (fromX + toX) / 2;

                // Thickness proportional to weight
                const thickness = 1 + weight * 4;
                const opacity = 0.3 + weight * 0.6;

                return (
                  <g key={`arc-${j}`}>
                    <path
                      d={`M ${fromX} ${fromY} Q ${midX} ${fromY - arcHeight} ${toX} ${toY}`}
                      fill="none"
                      stroke="var(--primary)"
                      strokeWidth={thickness}
                      opacity={opacity}
                      strokeLinecap="round"
                    />
                    {/* Arrow head at the target */}
                    <circle
                      cx={toX} cy={toY}
                      r={2 + weight * 2}
                      fill="var(--primary)"
                      opacity={opacity}
                    />
                  </g>
                );
              })}

              {/* Self-attention arrow (token attending to itself) */}
              {selectedToken !== null && weights[selectedToken]?.[selectedToken] > 0.03 && (
                <g>
                  <path
                    d={`M ${40 + selectedToken * 70 + 15} ${128} Q ${40 + selectedToken * 70 + 25} ${108} ${40 + selectedToken * 70} ${108} Q ${40 + selectedToken * 70 - 25} ${108} ${40 + selectedToken * 70 - 15} ${128}`}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth={1 + weights[selectedToken][selectedToken] * 3}
                    opacity={0.5}
                    strokeLinecap="round"
                  />
                  <text
                    x={40 + selectedToken * 70}
                    y={100}
                    textAnchor="middle"
                    fill="var(--accent)"
                    fontSize="8"
                    fontFamily="var(--font-mono)"
                  >
                    self: {(weights[selectedToken][selectedToken] * 100).toFixed(0)}%
                  </text>
                </g>
              )}

              {/* Label */}
              {selectedToken !== null && (
                <text
                  x={40 + selectedToken * 70}
                  y={12}
                  textAnchor="middle"
                  fill="var(--primary)"
                  fontSize="9"
                  fontFamily="var(--font-mono)"
                >
                  query: &quot;{tokens[selectedToken]}&quot;
                </text>
              )}
            </svg>
          </div>

          {/* Token row for longer sentences (fallback) */}
          {tokens.length > 14 && (
            <div className="flex flex-wrap gap-2 mb-2">
              {tokens.map((token, i) => {
                const isSelected = selectedToken === i;
                const attendedWeight = selectedToken !== null ? weights[selectedToken]?.[i] || 0 : 0;
                const isAttended = selectedToken !== null && attendedWeight > 0.05;

                return (
                  <button
                    key={i}
                    onClick={() => setSelectedToken(selectedToken === i ? null : i)}
                    className="relative rounded px-2.5 py-1.5 font-mono text-xs transition-all"
                    style={{
                      backgroundColor: isSelected ? "var(--primary)" : isAttended ? `rgba(232,151,108,${attendedWeight * 0.6})` : "var(--bg)",
                      color: isSelected ? "white" : "var(--text-secondary)",
                      border: isSelected ? "2px solid var(--primary)" : isAttended ? "2px solid var(--primary)" : "1px solid var(--border)",
                    }}
                  >
                    {token}
                  </button>
                );
              })}
            </div>
          )}

          {/* Attention weights bar for selected token */}
          {selectedToken !== null && weights[selectedToken] && (
            <div className="mt-4 space-y-1.5">
              {tokens.map((token, j) => {
                if (causal && j > selectedToken) return null;
                const weight = weights[selectedToken][j];
                return (
                  <div key={j} className="flex items-center gap-2">
                    <span
                      className="w-16 shrink-0 text-right font-mono text-xs truncate"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {token}
                    </span>
                    <div className="flex-1 h-4 rounded overflow-hidden" style={{ backgroundColor: "var(--bg)" }}>
                      <div
                        className="h-full rounded transition-all duration-300"
                        style={{
                          width: `${(weight / maxWeight) * 100}%`,
                          backgroundColor: "var(--primary)",
                          opacity: 0.3 + weight * 0.7,
                        }}
                      />
                    </div>
                    <span className="w-12 shrink-0 font-mono text-xs text-right" style={{ color: "var(--text-muted)" }}>
                      {(weight * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Attention heatmap */}
      {tokens.length > 0 && tokens.length <= 12 && (
        <div className="mb-4">
          <p className="mb-2 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            full attention matrix (row = query token, column = key token):
          </p>
          <div className="overflow-x-auto">
            <div className="inline-block">
              {/* Column headers */}
              <div className="flex">
                <div className="w-16 shrink-0" />
                {tokens.map((token, j) => (
                  <div
                    key={j}
                    className="w-10 shrink-0 text-center font-mono truncate"
                    style={{ fontSize: "8px", color: "var(--text-muted)" }}
                  >
                    {token.slice(0, 4)}
                  </div>
                ))}
              </div>

              {/* Rows */}
              {weights.map((row, i) => (
                <div key={i} className="flex items-center">
                  <div
                    className="w-16 shrink-0 text-right pr-2 font-mono truncate"
                    style={{ fontSize: "8px", color: selectedToken === i ? "var(--primary)" : "var(--text-muted)" }}
                  >
                    {tokens[i]}
                  </div>
                  {row.map((weight, j) => (
                    <div
                      key={j}
                      className="w-10 h-8 shrink-0 flex items-center justify-center cursor-pointer border"
                      style={{
                        backgroundColor: (causal && j > i)
                          ? "var(--bg-elevated)"
                          : `rgba(232,151,108,${weight * 0.8})`,
                        borderColor: "var(--border)",
                        opacity: (causal && j > i) ? 0.3 : 1,
                      }}
                      onClick={() => setSelectedToken(i)}
                    >
                      {!(causal && j > i) && (
                        <span className="font-mono" style={{ fontSize: "7px", color: weight > 0.4 ? "white" : "var(--text-muted)" }}>
                          {(weight * 100).toFixed(0)}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <p className="mt-2 font-mono" style={{ fontSize: "9px", color: "var(--text-muted)" }}>
            darker = higher attention weight{causal && " · gray cells = masked (future tokens)"} · {causal ? "lower triangle only (causal)" : "full matrix (bidirectional)"}
          </p>
        </div>
      )}

      {/* Note about simulation */}
      <div
        className="rounded p-3"
        style={{ borderLeft: "3px solid var(--accent)", backgroundColor: "rgba(212,168,67,0.04)" }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          Note: this uses random Q, K projections for demonstration — the attention patterns here aren&apos;t linguistically meaningful. In a trained model, the learned W<sub>Q</sub> and W<sub>K</sub> matrices produce patterns where tokens attend to grammatically and semantically relevant tokens (subjects attend to verbs, pronouns attend to their referents, etc).
        </p>
      </div>
    </div>
  );
}
