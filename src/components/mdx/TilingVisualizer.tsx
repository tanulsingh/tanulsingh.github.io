"use client";

import { useState, useMemo, useCallback } from "react";

const CORAL = "#E8976C";
const SAGE = "#8BAF7A";
const INDIGO = "#7B68EE";
const AMBER = "#D4A843";
const MUTED = "#94a3b8";
const TEAL = "#45B7A0";

const GRID = 4;
const TILE = 2;
const TILES_PER_DIM = GRID / TILE;

type TileCoord = { r: number; c: number };

function buildNaiveSteps() {
  const steps: { aRow: number; bCol: number; desc: string }[] = [];
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      steps.push({
        aRow: i,
        bCol: j,
        desc: `Load full row ${i} of A and full column ${j} of B to compute C[${i},${j}]`,
      });
    }
  }
  return steps;
}

function buildTiledSteps() {
  const steps: {
    aTile: TileCoord;
    bTile: TileCoord;
    cTile: TileCoord;
    kStep: number;
    desc: string;
  }[] = [];
  for (let bi = 0; bi < TILES_PER_DIM; bi++) {
    for (let bj = 0; bj < TILES_PER_DIM; bj++) {
      for (let bk = 0; bk < TILES_PER_DIM; bk++) {
        steps.push({
          aTile: { r: bi, c: bk },
          bTile: { r: bk, c: bj },
          cTile: { r: bi, c: bj },
          kStep: bk,
          desc: `Load A[${bi},${bk}] and B[${bk},${bj}] into SRAM → accumulate into C[${bi},${bj}]`,
        });
      }
    }
  }
  return steps;
}

function MatrixGrid({
  label,
  color,
  highlightCells,
  highlightTile,
  dimLabel,
}: {
  label: string;
  color: string;
  highlightCells?: { rows: number[]; cols: number[] };
  highlightTile?: TileCoord;
  dimLabel?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="font-mono text-xs font-semibold" style={{ color }}>
        {label}
      </span>
      <div
        className="grid gap-px rounded border p-0.5"
        style={{
          gridTemplateColumns: `repeat(${GRID}, 1fr)`,
          borderColor: `${color}60`,
          backgroundColor: `${color}08`,
        }}
      >
        {Array.from({ length: GRID * GRID }).map((_, idx) => {
          const r = Math.floor(idx / GRID);
          const c = idx % GRID;

          let isHighlighted = false;
          if (highlightCells) {
            isHighlighted =
              highlightCells.rows.includes(r) && highlightCells.cols.includes(c);
          }
          if (highlightTile) {
            const tr = highlightTile.r * TILE;
            const tc = highlightTile.c * TILE;
            isHighlighted = r >= tr && r < tr + TILE && c >= tc && c < tc + TILE;
          }

          return (
            <div
              key={idx}
              style={{
                width: 28,
                height: 28,
                borderRadius: 3,
                backgroundColor: isHighlighted
                  ? `${color}90`
                  : `${color}18`,
                transition: "background-color 0.2s",
              }}
            />
          );
        })}
      </div>
      {dimLabel && (
        <span className="font-mono text-xs" style={{ color: MUTED }}>
          {dimLabel}
        </span>
      )}
    </div>
  );
}

function MemoryBox({
  label,
  color,
  items,
  capacity,
}: {
  label: string;
  color: string;
  items: string[];
  capacity: string;
}) {
  return (
    <div
      className="rounded-lg border p-3 min-w-[120px]"
      style={{ borderColor: `${color}60`, backgroundColor: `${color}08` }}
    >
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="font-mono text-xs font-semibold" style={{ color }}>
          {label}
        </span>
      </div>
      <div className="text-xs" style={{ color: MUTED }}>
        {capacity}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {items.map((item, i) => (
          <span
            key={i}
            className="font-mono text-xs px-1.5 py-0.5 rounded"
            style={{ backgroundColor: `${color}25`, color }}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function NaiveView({ step }: { step: number }) {
  const steps = useMemo(() => buildNaiveSteps(), []);
  const current = steps[step];

  const allRows = Array.from({ length: GRID }, (_, i) => i);
  const allCols = Array.from({ length: GRID }, (_, i) => i);

  return (
    <div>
      <div className="flex items-center justify-center gap-4 flex-wrap">
        <MatrixGrid
          label="A"
          color={CORAL}
          dimLabel="4×4"
          highlightCells={{ rows: [current.aRow], cols: allCols }}
        />
        <span className="font-mono text-lg" style={{ color: MUTED }}>
          ×
        </span>
        <MatrixGrid
          label="B"
          color={INDIGO}
          dimLabel="4×4"
          highlightCells={{ rows: allRows, cols: [current.bCol] }}
        />
        <span className="font-mono text-lg" style={{ color: MUTED }}>
          =
        </span>
        <MatrixGrid
          label="C"
          color={SAGE}
          dimLabel="4×4"
          highlightCells={{ rows: [current.aRow], cols: [current.bCol] }}
        />
      </div>

      <div className="mt-4 flex gap-3 justify-center flex-wrap">
        <MemoryBox
          label="HBM"
          color={CORAL}
          items={["A (full)", "B (full)", "C (full)"]}
          capacity="Large, slow (~2 TB/s)"
        />
        <MemoryBox
          label="SRAM"
          color={TEAL}
          items={[`Row ${current.aRow} of A`, `Col ${current.bCol} of B`]}
          capacity="Small, fast (~19 TB/s)"
        />
      </div>

      <div
        className="mt-3 text-xs text-center"
        style={{ color: MUTED }}
      >
        {current.desc}
      </div>
      <div
        className="mt-1 text-xs text-center font-mono"
        style={{ color: CORAL }}
      >
        HBM loads so far: {(step + 1) * 2 * GRID} elements (row + column each time)
      </div>
    </div>
  );
}

function TiledView({ step }: { step: number }) {
  const steps = useMemo(() => buildTiledSteps(), []);
  const current = steps[step];

  const completedCTiles = new Set<string>();
  for (let s = 0; s <= step; s++) {
    const st = steps[s];
    if (st.kStep === TILES_PER_DIM - 1) {
      completedCTiles.add(`${st.cTile.r},${st.cTile.c}`);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-center gap-4 flex-wrap">
        <MatrixGrid
          label="A"
          color={CORAL}
          dimLabel="4×4 (2×2 tiles)"
          highlightTile={current.aTile}
        />
        <span className="font-mono text-lg" style={{ color: MUTED }}>
          ×
        </span>
        <MatrixGrid
          label="B"
          color={INDIGO}
          dimLabel="4×4 (2×2 tiles)"
          highlightTile={current.bTile}
        />
        <span className="font-mono text-lg" style={{ color: MUTED }}>
          =
        </span>
        <MatrixGrid
          label="C"
          color={SAGE}
          dimLabel="4×4 (2×2 tiles)"
          highlightTile={current.cTile}
        />
      </div>

      <div className="mt-4 flex gap-3 justify-center flex-wrap">
        <MemoryBox
          label="HBM"
          color={CORAL}
          items={["A (full)", "B (full)", "C (partial)"]}
          capacity="Large, slow (~2 TB/s)"
        />
        <MemoryBox
          label="SRAM"
          color={TEAL}
          items={[
            `A[${current.aTile.r},${current.aTile.c}]`,
            `B[${current.bTile.r},${current.bTile.c}]`,
            `C[${current.cTile.r},${current.cTile.c}] partial`,
          ]}
          capacity="Small, fast (~19 TB/s)"
        />
      </div>

      <div
        className="mt-3 text-xs text-center"
        style={{ color: MUTED }}
      >
        {current.desc}
      </div>
      <div
        className="mt-1 text-xs text-center font-mono"
        style={{ color: SAGE }}
      >
        HBM loads so far: {(step + 1) * 2 * TILE * TILE} elements (one tile of A + one tile of B each time)
      </div>
    </div>
  );
}

export function TilingVisualizer() {
  const [mode, setMode] = useState<"naive" | "tiled">("naive");
  const [step, setStep] = useState(0);

  const naiveSteps = useMemo(() => buildNaiveSteps(), []);
  const tiledSteps = useMemo(() => buildTiledSteps(), []);
  const maxStep = mode === "naive" ? naiveSteps.length - 1 : tiledSteps.length - 1;

  const handleModeSwitch = useCallback(
    (m: "naive" | "tiled") => {
      setMode(m);
      setStep(0);
    },
    []
  );

  const naiveTotalLoads = naiveSteps.length * 2 * GRID;
  const tiledTotalLoads = tiledSteps.length * 2 * TILE * TILE;

  return (
    <div
      className="my-8 rounded-xl border p-5"
      style={{
        borderColor: "var(--border)",
        backgroundColor: "var(--bg-card)",
      }}
    >
      <div className="mb-1 text-base font-bold" style={{ color: "var(--text-primary)" }}>
        Matrix Multiply: Naive vs Tiled
      </div>
      <div className="mb-4 text-sm" style={{ color: MUTED }}>
        Step through both approaches to see how tiling reduces data movement between
        HBM (slow) and SRAM (fast). Same arithmetic, far fewer memory loads.
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2 mb-4">
        {(["naive", "tiled"] as const).map((m) => (
          <button
            key={m}
            onClick={() => handleModeSwitch(m)}
            className="px-3 py-1.5 rounded-md font-mono text-xs font-semibold transition-colors"
            style={{
              backgroundColor:
                mode === m ? (m === "naive" ? `${CORAL}20` : `${SAGE}20`) : "transparent",
              color: mode === m ? (m === "naive" ? CORAL : SAGE) : MUTED,
              border: `1px solid ${mode === m ? (m === "naive" ? CORAL : SAGE) : `${MUTED}40`}`,
            }}
          >
            {m === "naive" ? "Naive (element-by-element)" : "Tiled (block-by-block)"}
          </button>
        ))}
      </div>

      {/* Visualization */}
      {mode === "naive" ? <NaiveView step={step} /> : <TiledView step={step} />}

      {/* Step controls */}
      <div className="mt-4 flex items-center justify-center gap-3">
        <button
          onClick={() => setStep(0)}
          disabled={step === 0}
          className="px-2 py-1 rounded text-xs font-mono"
          style={{
            color: step === 0 ? `${MUTED}40` : MUTED,
            border: `1px solid ${MUTED}30`,
          }}
        >
          ⟨⟨
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
        <span className="font-mono text-xs" style={{ color: MUTED }}>
          Step {step + 1} / {maxStep + 1}
        </span>
        <button
          onClick={() => setStep(Math.min(maxStep, step + 1))}
          disabled={step === maxStep}
          className="px-2 py-1 rounded text-xs font-mono"
          style={{
            color: step === maxStep ? `${MUTED}40` : MUTED,
            border: `1px solid ${MUTED}30`,
          }}
        >
          Next ⟩
        </button>
        <button
          onClick={() => setStep(maxStep)}
          disabled={step === maxStep}
          className="px-2 py-1 rounded text-xs font-mono"
          style={{
            color: step === maxStep ? `${MUTED}40` : MUTED,
            border: `1px solid ${MUTED}30`,
          }}
        >
          ⟩⟩
        </button>
      </div>

      {/* Comparison stats */}
      <div
        className="mt-4 rounded-lg border p-3 grid grid-cols-2 gap-4 text-xs"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-elevated)" }}
      >
        <div>
          <div className="font-semibold mb-1" style={{ color: CORAL }}>
            Naive Approach
          </div>
          <div style={{ color: MUTED }}>
            Steps: {naiveSteps.length} (one per output element)
          </div>
          <div style={{ color: MUTED }}>
            Total HBM loads: <span className="font-mono font-semibold" style={{ color: CORAL }}>{naiveTotalLoads}</span> elements
          </div>
          <div style={{ color: MUTED }}>
            Each row/column reloaded {GRID} times
          </div>
        </div>
        <div>
          <div className="font-semibold mb-1" style={{ color: SAGE }}>
            Tiled Approach
          </div>
          <div style={{ color: MUTED }}>
            Steps: {tiledSteps.length} (one per tile pair)
          </div>
          <div style={{ color: MUTED }}>
            Total HBM loads: <span className="font-mono font-semibold" style={{ color: SAGE }}>{tiledTotalLoads}</span> elements
          </div>
          <div style={{ color: MUTED }}>
            Each tile loaded {TILES_PER_DIM} times (once per k-step)
          </div>
        </div>
      </div>
    </div>
  );
}
