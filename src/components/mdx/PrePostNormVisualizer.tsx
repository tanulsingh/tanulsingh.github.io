"use client";

import { useState } from "react";

const CORAL = "#E8976C";
const SAGE = "#8BAF7A";
const INDIGO = "#7B68EE";
const TEAL = "#45B7A0";
const MUTED = "#94a3b8";
const AMBER = "#D4A843";

type Mode = "pre" | "post";

function Arrow({ color, height = 20 }: { color: string; height?: number }) {
  return (
    <div className="flex flex-col items-center" style={{ height: height + 6 }}>
      <div style={{ width: 2, height, backgroundColor: color }} />
      <div
        style={{
          width: 0,
          height: 0,
          borderLeft: "4px solid transparent",
          borderRight: "4px solid transparent",
          borderTop: `6px solid ${color}`,
        }}
      />
    </div>
  );
}

function Block({
  label,
  color,
  subtitle,
  width = 90,
}: {
  label: string;
  color: string;
  subtitle?: string;
  width?: number;
}) {
  return (
    <div
      className="rounded-lg px-3 py-2 text-center font-mono text-xs font-semibold"
      style={{
        backgroundColor: `${color}18`,
        border: `2px solid ${color}`,
        color,
        width,
      }}
    >
      {label}
      {subtitle && (
        <div style={{ fontSize: 9, fontWeight: 400, marginTop: 2, opacity: 0.8 }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}

function PostNormDiagram() {
  const mainColor = CORAL;

  return (
    <div className="flex flex-col items-center">
      {/* Title */}
      <div
        className="font-mono text-sm font-bold mb-1 px-3 py-1 rounded"
        style={{ color: mainColor, backgroundColor: `${mainColor}12` }}
      >
        Post-Norm
      </div>
      <div className="font-mono text-xs mb-3" style={{ color: MUTED }}>
        Norm(x + f(x))
      </div>

      {/* Input */}
      <Block label="x" color={AMBER} subtitle="input" />

      {/* Fork: single vertical line then horizontal split */}
      <div style={{ width: 2, height: 12, backgroundColor: mainColor }} />
      <div className="flex items-start">
        {/* Horizontal bar */}
        <div style={{ width: 140, height: 2, backgroundColor: mainColor }} />
      </div>

      {/* Two paths side by side */}
      <div className="flex" style={{ width: 140 }}>
        {/* Left: branch */}
        <div className="flex flex-col items-center" style={{ width: 70 }}>
          <div style={{ width: 2, height: 12, backgroundColor: INDIGO }} />
          <div
            className="font-mono mb-1"
            style={{ fontSize: 9, color: MUTED }}
          >
            branch
          </div>
          <Block label="f(x)" color={INDIGO} subtitle="sublayer" width={70} />
          <Arrow color={INDIGO} height={14} />
        </div>

        {/* Right: skip */}
        <div className="flex flex-col items-center" style={{ width: 70 }}>
          <div style={{ width: 2, height: 12, backgroundColor: mainColor }} />
          <div
            className="font-mono mb-1"
            style={{ fontSize: 9, color: MUTED }}
          >
            skip
          </div>
          <div style={{ width: 2, height: 48, backgroundColor: mainColor }} />
          <div
            style={{
              width: 0,
              height: 0,
              borderLeft: "4px solid transparent",
              borderRight: "4px solid transparent",
              borderTop: `6px solid ${mainColor}`,
            }}
          />
        </div>
      </div>

      {/* Merge bar */}
      <div className="flex items-start">
        <div style={{ width: 140, height: 2, backgroundColor: mainColor }} />
      </div>
      <div style={{ width: 2, height: 8, backgroundColor: mainColor }} />

      {/* Add node */}
      <div
        className="rounded-full flex items-center justify-center font-mono font-bold"
        style={{
          width: 26,
          height: 26,
          border: `2px solid ${SAGE}`,
          color: SAGE,
          fontSize: 14,
          backgroundColor: `${SAGE}15`,
        }}
      >
        +
      </div>
      <div className="font-mono" style={{ fontSize: 9, color: MUTED, marginTop: 2 }}>
        x + f(x)
      </div>

      <Arrow color={mainColor} height={10} />

      {/* Norm — ON THE HIGHWAY */}
      <Block label="Norm" color={mainColor} subtitle="⚠ on highway!" width={90} />

      <Arrow color={mainColor} height={10} />

      {/* Output */}
      <Block label="output" color={AMBER} />

      {/* Gradient note */}
      <div
        className="mt-4 rounded-lg border p-3 text-center"
        style={{
          borderColor: `${mainColor}40`,
          backgroundColor: `${mainColor}08`,
          maxWidth: 200,
        }}
      >
        <div className="font-mono text-xs font-semibold" style={{ color: mainColor }}>
          Gradient path
        </div>
        <div className="font-mono mt-1" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
          ∂out/∂x passes through J<sub>norm</sub>
        </div>
        <div className="font-mono mt-1" style={{ fontSize: 9, color: MUTED }}>
          N layers → N norm Jacobians compound
        </div>
      </div>
    </div>
  );
}

function PreNormDiagram() {
  const mainColor = SAGE;

  return (
    <div className="flex flex-col items-center">
      {/* Title */}
      <div
        className="font-mono text-sm font-bold mb-1 px-3 py-1 rounded"
        style={{ color: mainColor, backgroundColor: `${mainColor}12` }}
      >
        Pre-Norm
      </div>
      <div className="font-mono text-xs mb-3" style={{ color: MUTED }}>
        x + f(Norm(x))
      </div>

      {/* Input */}
      <Block label="x" color={AMBER} subtitle="input" />

      {/* Fork */}
      <div style={{ width: 2, height: 12, backgroundColor: mainColor }} />
      <div className="flex items-start">
        <div style={{ width: 160, height: 2, backgroundColor: mainColor }} />
      </div>

      {/* Two paths side by side */}
      <div className="flex" style={{ width: 160 }}>
        {/* Left: branch (Norm then sublayer) */}
        <div className="flex flex-col items-center" style={{ width: 80 }}>
          <div style={{ width: 2, height: 12, backgroundColor: TEAL }} />
          <div
            className="font-mono mb-1"
            style={{ fontSize: 9, color: MUTED }}
          >
            branch
          </div>
          <Block label="Norm" color={TEAL} subtitle="only here" width={74} />
          <Arrow color={INDIGO} height={8} />
          <Block label="f(·)" color={INDIGO} subtitle="sublayer" width={74} />
          <Arrow color={INDIGO} height={8} />
        </div>

        {/* Right: skip — CLEAN */}
        <div className="flex flex-col items-center" style={{ width: 80 }}>
          <div style={{ width: 2, height: 12, backgroundColor: mainColor }} />
          <div
            className="font-mono mb-1 font-bold"
            style={{ fontSize: 9, color: mainColor }}
          >
            skip (clean!)
          </div>
          <div style={{ width: 2, height: 100, backgroundColor: mainColor }} />
          <div
            style={{
              width: 0,
              height: 0,
              borderLeft: "4px solid transparent",
              borderRight: "4px solid transparent",
              borderTop: `6px solid ${mainColor}`,
            }}
          />
        </div>
      </div>

      {/* Merge bar */}
      <div className="flex items-start">
        <div style={{ width: 160, height: 2, backgroundColor: mainColor }} />
      </div>
      <div style={{ width: 2, height: 8, backgroundColor: mainColor }} />

      {/* Add node */}
      <div
        className="rounded-full flex items-center justify-center font-mono font-bold"
        style={{
          width: 26,
          height: 26,
          border: `2px solid ${SAGE}`,
          color: SAGE,
          fontSize: 14,
          backgroundColor: `${SAGE}15`,
        }}
      >
        +
      </div>
      <div className="font-mono" style={{ fontSize: 9, color: MUTED, marginTop: 2 }}>
        x + f(Norm(x))
      </div>

      <Arrow color={mainColor} height={10} />

      {/* Output */}
      <Block label="output" color={AMBER} />

      {/* Gradient note */}
      <div
        className="mt-4 rounded-lg border p-3 text-center"
        style={{
          borderColor: `${mainColor}40`,
          backgroundColor: `${mainColor}08`,
          maxWidth: 200,
        }}
      >
        <div className="font-mono text-xs font-semibold" style={{ color: mainColor }}>
          Gradient path
        </div>
        <div className="font-mono mt-1" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
          ∂out/∂x = I + branch terms
        </div>
        <div className="font-mono mt-1" style={{ fontSize: 9, color: MUTED }}>
          Identity survives at any depth!
        </div>
      </div>
    </div>
  );
}

export function PrePostNormVisualizer() {
  const [tab, setTab] = useState<"both" | Mode>("both");

  return (
    <div
      className="my-8 rounded-xl border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-card)" }}
    >
      <div className="mb-1 text-base font-bold" style={{ color: "var(--text-primary)" }}>
        Pre-Norm vs Post-Norm: Where Does Norm Sit?
      </div>
      <div className="mb-4 text-sm" style={{ color: MUTED }}>
        One layer, two architectures. The only difference: whether Norm is on the skip highway or tucked inside the branch.
      </div>

      {/* Tab switcher */}
      <div className="flex gap-2 mb-5">
        {(["both", "post", "pre"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className="px-3 py-1 rounded font-mono text-xs"
            style={{
              backgroundColor: tab === t ? `${INDIGO}20` : "transparent",
              border: `1px solid ${tab === t ? INDIGO : `${MUTED}30`}`,
              color: tab === t ? INDIGO : MUTED,
              fontWeight: tab === t ? 700 : 400,
            }}
          >
            {t === "both" ? "Side by Side" : t === "post" ? "Post-Norm" : "Pre-Norm"}
          </button>
        ))}
      </div>

      {/* Diagrams */}
      <div className="flex justify-center gap-16 flex-wrap">
        {(tab === "both" || tab === "post") && <PostNormDiagram />}
        {(tab === "both" || tab === "pre") && <PreNormDiagram />}
      </div>

      {/* Key insight */}
      <div
        className="mt-5 rounded-lg border p-3 text-sm"
        style={{
          borderColor: "var(--border)",
          backgroundColor: "var(--bg-elevated)",
          color: "var(--text-primary)",
        }}
      >
        <span className="font-semibold" style={{ color: SAGE }}>Key insight:</span>{" "}
        In post-norm, the gradient from output to input <em>must</em> pass through Norm — there&apos;s no alternative path.
        In pre-norm, the skip connection gives a direct additive path that bypasses Norm entirely.
        Stack N layers and post-norm forces N norm Jacobians on the gradient; pre-norm always preserves an identity term.
      </div>
    </div>
  );
}
