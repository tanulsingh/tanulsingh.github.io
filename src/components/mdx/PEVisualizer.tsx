"use client";

import { useState, useMemo } from "react";

function computePE(pos: number, dModel: number): number[] {
  const pe: number[] = [];
  for (let i = 0; i < dModel; i++) {
    const dimIdx = Math.floor(i / 2);
    const freq = 1 / Math.pow(10000, (2 * dimIdx) / dModel);
    if (i % 2 === 0) {
      pe.push(Math.sin(freq * pos));
    } else {
      pe.push(Math.cos(freq * pos));
    }
  }
  return pe;
}

function PEBar({ value, dim, isHighlighted }: { value: number; dim: number; isHighlighted: boolean }) {
  const width = Math.abs(value) * 100;
  const isPositive = value >= 0;

  return (
    <div className="flex items-center gap-1" style={{ height: "8px" }}>
      <div className="flex w-full items-center" style={{ height: "100%" }}>
        {/* Negative side */}
        <div className="flex h-full w-1/2 justify-end">
          {!isPositive && (
            <div
              className="h-full rounded-l-sm"
              style={{
                width: `${width}%`,
                backgroundColor: isHighlighted ? "var(--sage)" : "var(--primary)",
                opacity: isHighlighted ? 0.8 : 0.4,
              }}
            />
          )}
        </div>
        {/* Center line */}
        <div className="h-full w-px" style={{ backgroundColor: "var(--border)" }} />
        {/* Positive side */}
        <div className="flex h-full w-1/2">
          {isPositive && (
            <div
              className="h-full rounded-r-sm"
              style={{
                width: `${width}%`,
                backgroundColor: isHighlighted ? "var(--sage)" : "var(--primary)",
                opacity: isHighlighted ? 0.8 : 0.4,
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export function PEVisualizer() {
  const [pos1, setPos1] = useState(3);
  const [pos2, setPos2] = useState(50);
  const [showDims, setShowDims] = useState(64);

  const dModel = 128;

  const pe1 = useMemo(() => computePE(pos1, dModel), [pos1]);
  const pe2 = useMemo(() => computePE(pos2, dModel), [pos2]);

  // Compute which dimensions differ most
  const diffs = pe1.map((v, i) => Math.abs(v - pe2[i]));
  const maxDiff = Math.max(...diffs);

  // Dot product (measures similarity)
  const dotProduct = pe1.reduce((sum, v, i) => sum + v * pe2[i], 0);

  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        interactive
      </p>
      <h4 className="mb-4 text-lg font-bold" style={{ fontFamily: "var(--font-serif)" }}>
        Positional Encoding Visualizer
      </h4>

      {/* Position inputs */}
      <div className="mb-4 flex flex-wrap gap-6">
        <div>
          <label className="mb-1 block font-mono text-xs" style={{ color: "var(--primary)" }}>
            Position A:
          </label>
          <input
            type="range"
            min={0}
            max={200}
            value={pos1}
            onChange={(e) => setPos1(Number(e.target.value))}
            className="w-32"
          />
          <span className="ml-2 font-mono text-sm" style={{ color: "var(--text)" }}>
            {pos1}
          </span>
        </div>
        <div>
          <label className="mb-1 block font-mono text-xs" style={{ color: "var(--sage)" }}>
            Position B:
          </label>
          <input
            type="range"
            min={0}
            max={200}
            value={pos2}
            onChange={(e) => setPos2(Number(e.target.value))}
            className="w-32"
          />
          <span className="ml-2 font-mono text-sm" style={{ color: "var(--text)" }}>
            {pos2}
          </span>
        </div>
      </div>

      {/* Summary stats */}
      <div
        className="mb-4 rounded p-3"
        style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
      >
        <div className="flex flex-wrap gap-6 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          <span>distance: <span style={{ color: "var(--text-secondary)" }}>{Math.abs(pos2 - pos1)} positions</span></span>
          <span>dot product: <span style={{ color: "var(--text-secondary)" }}>{dotProduct.toFixed(2)}</span></span>
          <span>
            similarity: <span style={{ color: dotProduct > dModel * 0.5 ? "var(--sage)" : "var(--primary)" }}>
              {dotProduct > dModel * 0.7 ? "high (nearby)" : dotProduct > dModel * 0.3 ? "medium" : "low (far apart)"}
            </span>
          </span>
        </div>
      </div>

      {/* Dimension range control */}
      <div className="mb-3 flex items-center gap-3">
        <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>show dimensions:</span>
        {[32, 64, 128].map((n) => (
          <button
            key={n}
            onClick={() => setShowDims(n)}
            className="rounded px-2 py-0.5 font-mono text-xs"
            style={{
              backgroundColor: showDims === n ? "var(--tag-bg)" : "transparent",
              color: showDims === n ? "var(--primary)" : "var(--text-muted)",
            }}
          >
            {n}
          </button>
        ))}
      </div>

      {/* Visualization */}
      <div className="mb-4 overflow-hidden rounded border" style={{ borderColor: "var(--border)" }}>
        {/* Header */}
        <div className="flex font-mono text-xs" style={{ backgroundColor: "var(--bg-elevated)" }}>
          <div className="w-12 shrink-0 px-2 py-1" style={{ color: "var(--text-muted)" }}>dim</div>
          <div className="w-14 shrink-0 px-1 py-1 text-center" style={{ color: "var(--text-muted)" }}>freq</div>
          <div className="flex-1 px-2 py-1 text-center" style={{ color: "var(--primary)" }}>pos {pos1}</div>
          <div className="flex-1 px-2 py-1 text-center" style={{ color: "var(--sage)" }}>pos {pos2}</div>
          <div className="w-14 shrink-0 px-1 py-1 text-center" style={{ color: "var(--text-muted)" }}>Δ</div>
        </div>

        {/* Rows — show every 2 dims (one sin/cos pair) */}
        <div className="max-h-72 overflow-y-auto">
          {Array.from({ length: Math.min(showDims / 2, dModel / 2) }).map((_, dimPair) => {
            const sinIdx = dimPair * 2;
            const freq = 1 / Math.pow(10000, (2 * dimPair) / dModel);
            const diff = Math.abs(pe1[sinIdx] - pe2[sinIdx]);
            const isHighFreq = dimPair < 8;
            const isLowFreq = dimPair > showDims / 2 - 8;

            return (
              <div
                key={dimPair}
                className="flex items-center border-t"
                style={{
                  borderColor: "var(--border)",
                  backgroundColor: isHighFreq ? "rgba(232,151,108,0.03)" : isLowFreq ? "rgba(139,175,122,0.03)" : "transparent",
                }}
              >
                <div className="w-12 shrink-0 px-2 py-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
                  {sinIdx}
                </div>
                <div className="w-14 shrink-0 px-1 py-1 text-center font-mono" style={{ color: "var(--text-muted)", fontSize: "9px" }}>
                  {freq > 0.01 ? freq.toFixed(3) : freq.toExponential(0)}
                </div>
                <div className="flex-1 px-1 py-1">
                  <PEBar value={pe1[sinIdx]} dim={sinIdx} isHighlighted={false} />
                </div>
                <div className="flex-1 px-1 py-1">
                  <PEBar value={pe2[sinIdx]} dim={sinIdx} isHighlighted={true} />
                </div>
                <div className="w-14 shrink-0 px-1 py-1 text-center font-mono" style={{ color: "var(--text-muted)", fontSize: "9px" }}>
                  {diff.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Insight */}
      <div
        className="rounded p-3"
        style={{ borderLeft: "3px solid var(--primary)", backgroundColor: "rgba(232,151,108,0.04)" }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          {Math.abs(pos2 - pos1) <= 3
            ? "Notice: nearby positions have nearly identical values in the low-frequency (bottom) dimensions, but differ clearly in the high-frequency (top) dimensions. The 'seconds hand' tells them apart."
            : Math.abs(pos2 - pos1) <= 20
            ? "The high-frequency dimensions (top rows) have cycled multiple times between these positions, while mid-frequency dimensions show a clear shift. Both scales contribute to distinguishing them."
            : "These positions are far apart. Even the slow-moving dimensions (bottom rows) show noticeable differences. The 'hour hand' is what distinguishes them — the 'seconds hand' has cycled many times and looks random."}
        </p>
      </div>
    </div>
  );
}
