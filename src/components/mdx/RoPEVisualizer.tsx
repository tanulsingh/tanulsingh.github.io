"use client";

import { useState, useMemo } from "react";

function rotateVec(x: number, y: number, angle: number): [number, number] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [x * cos - y * sin, x * sin + y * cos];
}

function Arrow({
  x1, y1, x2, y2, color, opacity = 1, dashed = false, label, labelOffset = 0,
}: {
  x1: number; y1: number; x2: number; y2: number;
  color: string; opacity?: number; dashed?: boolean;
  label?: string; labelOffset?: number;
}) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 8;
  const headAngle = Math.PI / 6;

  return (
    <g opacity={opacity}>
      <line
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={color} strokeWidth="2"
        strokeDasharray={dashed ? "4 3" : undefined}
      />
      <line
        x1={x2} y1={y2}
        x2={x2 - headLen * Math.cos(angle - headAngle)}
        y2={y2 - headLen * Math.sin(angle - headAngle)}
        stroke={color} strokeWidth="2"
      />
      <line
        x1={x2} y1={y2}
        x2={x2 - headLen * Math.cos(angle + headAngle)}
        y2={y2 - headLen * Math.sin(angle + headAngle)}
        stroke={color} strokeWidth="2"
      />
      {label && (
        <text
          x={x2 + 12 * Math.cos(angle + labelOffset)}
          y={y2 + 12 * Math.sin(angle + labelOffset)}
          fill={color} fontSize="11" fontFamily="var(--font-mono)"
          textAnchor="middle" dominantBaseline="middle"
        >
          {label}
        </text>
      )}
    </g>
  );
}

function ArcArrow({
  cx, cy, radius, startAngle, endAngle, color,
}: {
  cx: number; cy: number; radius: number;
  startAngle: number; endAngle: number; color: string;
}) {
  const start = { x: cx + radius * Math.cos(startAngle), y: cy + radius * Math.sin(startAngle) };
  const end = { x: cx + radius * Math.cos(endAngle), y: cy + radius * Math.sin(endAngle) };
  const largeArc = Math.abs(endAngle - startAngle) > Math.PI ? 1 : 0;
  const sweep = endAngle > startAngle ? 1 : 0;

  return (
    <path
      d={`M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} ${sweep} ${end.x} ${end.y}`}
      stroke={color} strokeWidth="1.5" fill="none" strokeDasharray="3 2" opacity="0.6"
    />
  );
}

export function RoPEVisualizer() {
  const [posI, setPosI] = useState(3);
  const [posJ, setPosJ] = useState(7);
  const [theta, setTheta] = useState(0.5);

  const cx = 180;
  const cy = 180;
  const vecLen = 80;

  // Original Q and K vectors (arbitrary fixed directions)
  const qAngle = -Math.PI / 6; // Q points slightly right-down
  const kAngle = Math.PI / 4;   // K points right-up

  const qOrig: [number, number] = [vecLen * Math.cos(qAngle), vecLen * Math.sin(qAngle)];
  const kOrig: [number, number] = [vecLen * Math.cos(kAngle), vecLen * Math.sin(kAngle)];

  // Rotated by position
  const rotI = theta * posI;
  const rotJ = theta * posJ;
  const qRot = rotateVec(qOrig[0], qOrig[1], rotI);
  const kRot = rotateVec(kOrig[0], kOrig[1], rotJ);

  // Dot product (proportional to cos of angle between rotated vectors)
  const dotProduct = qRot[0] * kRot[0] + qRot[1] * kRot[1];
  const normQ = Math.sqrt(qRot[0] ** 2 + qRot[1] ** 2);
  const normK = Math.sqrt(kRot[0] ** 2 + kRot[1] ** 2);
  const cosAngle = dotProduct / (normQ * normK);

  // Angle between the two rotated vectors
  const rotQAngle = Math.atan2(qRot[1], qRot[0]);
  const rotKAngle = Math.atan2(kRot[1], kRot[0]);

  // The offset
  const offset = posI - posJ;

  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        interactive
      </p>
      <h4 className="mb-4 text-lg font-bold" style={{ fontFamily: "var(--font-serif)" }}>
        RoPE — How Rotation Encodes Relative Position
      </h4>

      <div className="flex flex-col gap-6 md:flex-row">
        {/* SVG visualization */}
        <div className="flex-1">
          <svg viewBox="0 0 360 360" className="w-full max-w-sm mx-auto">
            {/* Grid circle */}
            <circle cx={cx} cy={cy} r={vecLen + 20} fill="none" stroke="var(--border)" strokeWidth="0.5" opacity="0.3" />
            <circle cx={cx} cy={cy} r={vecLen / 2} fill="none" stroke="var(--border)" strokeWidth="0.5" opacity="0.2" />

            {/* Axes */}
            <line x1={cx - vecLen - 30} y1={cy} x2={cx + vecLen + 30} y2={cy} stroke="var(--border)" strokeWidth="0.5" opacity="0.3" />
            <line x1={cx} y1={cy - vecLen - 30} x2={cx} y2={cy + vecLen + 30} stroke="var(--border)" strokeWidth="0.5" opacity="0.3" />

            {/* Original vectors (faded) */}
            <Arrow
              x1={cx} y1={cy} x2={cx + qOrig[0]} y2={cy + qOrig[1]}
              color="var(--primary)" opacity={0.25} dashed label="q" labelOffset={-0.5}
            />
            <Arrow
              x1={cx} y1={cy} x2={cx + kOrig[0]} y2={cy + kOrig[1]}
              color="var(--sage)" opacity={0.25} dashed label="k" labelOffset={0.5}
            />

            {/* Rotation arcs */}
            {Math.abs(rotI) > 0.05 && (
              <ArcArrow cx={cx} cy={cy} radius={35} startAngle={qAngle} endAngle={qAngle + rotI} color="var(--primary)" />
            )}
            {Math.abs(rotJ) > 0.05 && (
              <ArcArrow cx={cx} cy={cy} radius={45} startAngle={kAngle} endAngle={kAngle + rotJ} color="var(--sage)" />
            )}

            {/* Rotated vectors (bold) */}
            <Arrow
              x1={cx} y1={cy} x2={cx + qRot[0]} y2={cy + qRot[1]}
              color="var(--primary)" label={`R${posI}·q`} labelOffset={-0.4}
            />
            <Arrow
              x1={cx} y1={cy} x2={cx + kRot[0]} y2={cy + kRot[1]}
              color="var(--sage)" label={`R${posJ}·k`} labelOffset={0.4}
            />

            {/* Angle between rotated vectors */}
            <ArcArrow cx={cx} cy={cy} radius={60} startAngle={rotQAngle} endAngle={rotKAngle} color="var(--accent)" />

            {/* Center dot */}
            <circle cx={cx} cy={cy} r="3" fill="var(--text-muted)" />

            {/* Labels */}
            <text x={20} y={20} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
              dashed = original, solid = rotated
            </text>
          </svg>
        </div>

        {/* Controls & explanation */}
        <div className="flex-1 space-y-4">
          {/* Position controls */}
          <div>
            <label className="mb-1 block font-mono text-xs" style={{ color: "var(--primary)" }}>
              Query position (i): {posI}
            </label>
            <input
              type="range" min={0} max={20} value={posI}
              onChange={(e) => setPosI(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="mb-1 block font-mono text-xs" style={{ color: "var(--sage)" }}>
              Key position (j): {posJ}
            </label>
            <input
              type="range" min={0} max={20} value={posJ}
              onChange={(e) => setPosJ(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="mb-1 block font-mono text-xs" style={{ color: "var(--text-muted)" }}>
              θ<sub>i</sub> = 1/10000<sup>2i/d</sup> (frequency for this dimension pair): {theta.toFixed(2)}
            </label>
            <input
              type="range" min={0.1} max={1.5} step={0.05} value={theta}
              onChange={(e) => setTheta(Number(e.target.value))}
              className="w-full"
            />
            <p className="mt-1 font-mono" style={{ color: "var(--text-muted)", fontSize: "9px" }}>
              high θ = low dimension (fast rotation, local position) · low θ = high dimension (slow rotation, global position)
            </p>
          </div>

          {/* Results */}
          <div
            className="rounded p-3 font-mono text-xs"
            style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
          >
            <div className="space-y-1">
              <p style={{ color: "var(--text-muted)" }}>
                q rotated by: <span style={{ color: "var(--primary)" }}>θ×i = {theta.toFixed(2)}×{posI} = {(theta * posI).toFixed(2)} rad ({((theta * posI) * 180 / Math.PI).toFixed(1)}°)</span>
              </p>
              <p style={{ color: "var(--text-muted)" }}>
                k rotated by: <span style={{ color: "var(--sage)" }}>θ×j = {theta.toFixed(2)}×{posJ} = {(theta * posJ).toFixed(2)} rad ({((theta * posJ) * 180 / Math.PI).toFixed(1)}°)</span>
              </p>
              <p style={{ color: "var(--text-muted)" }}>
                offset (i−j): <span style={{ color: "var(--accent)" }}>{offset}</span> → angle difference: <span style={{ color: "var(--accent)" }}>{(theta * offset).toFixed(2)} rad ({((theta * offset) * 180 / Math.PI).toFixed(1)}°)</span>
              </p>
              <p style={{ color: "var(--text-muted)" }}>
                dot product: <span style={{ color: "var(--text)" }}>{dotProduct.toFixed(2)}</span>
              </p>
            </div>
          </div>

          {/* Rotation matrices */}
          <div
            className="rounded p-3 font-mono"
            style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)", fontSize: "10px" }}
          >
            <p className="mb-2 text-xs" style={{ color: "var(--text-muted)" }}>Rotation matrices applied:</p>
            <div className="mb-2">
              <p style={{ color: "var(--primary)" }}>R<sub>{posI}</sub> (for query at pos {posI}):</p>
              <p style={{ color: "var(--text-secondary)" }}>
                ┌ cos({(rotI).toFixed(2)})  −sin({(rotI).toFixed(2)}) ┐ &nbsp; ┌ {Math.cos(rotI).toFixed(3)} &nbsp;{(-Math.sin(rotI)).toFixed(3)} ┐
              </p>
              <p style={{ color: "var(--text-secondary)" }}>
                └ sin({(rotI).toFixed(2)}) &nbsp;&nbsp;cos({(rotI).toFixed(2)}) ┘ = └ {Math.sin(rotI).toFixed(3)} &nbsp;&nbsp;{Math.cos(rotI).toFixed(3)} ┘
              </p>
            </div>
            <div>
              <p style={{ color: "var(--sage)" }}>R<sub>{posJ}</sub> (for key at pos {posJ}):</p>
              <p style={{ color: "var(--text-secondary)" }}>
                ┌ cos({(rotJ).toFixed(2)})  −sin({(rotJ).toFixed(2)}) ┐ &nbsp; ┌ {Math.cos(rotJ).toFixed(3)} &nbsp;{(-Math.sin(rotJ)).toFixed(3)} ┐
              </p>
              <p style={{ color: "var(--text-secondary)" }}>
                └ sin({(rotJ).toFixed(2)}) &nbsp;&nbsp;cos({(rotJ).toFixed(2)}) ┘ = └ {Math.sin(rotJ).toFixed(3)} &nbsp;&nbsp;{Math.cos(rotJ).toFixed(3)} ┘
              </p>
            </div>
          </div>

          {/* Effective relative rotation */}
          <div
            className="rounded p-3 font-mono"
            style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)", fontSize: "10px" }}
          >
            <p className="mb-1" style={{ color: "var(--accent)" }}>
              Effective relative rotation R<sub>{offset}</sub>:
            </p>
            <p style={{ color: "var(--text-secondary)" }}>
              angle = θ×(i−j) = {theta.toFixed(2)}×{offset} = <strong style={{ color: "var(--accent)" }}>{(theta * offset).toFixed(2)} rad</strong>
            </p>
            <p className="mt-1" style={{ color: "var(--text-muted)" }}>
              (R<sub>i</sub>·q)·(R<sub>j</sub>·k) = q·R<sub>i−j</sub>·k → only offset matters!
            </p>
          </div>

          {/* Key insight */}
          <div
            className="rounded p-3"
            style={{ borderLeft: "3px solid var(--primary)", backgroundColor: "rgba(232,151,108,0.04)" }}
          >
            <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
              Try this: set i=3, j=7 (offset = −4). Note the dot product. Now set i=10, j=14 (same offset = −4). <strong style={{ color: "var(--primary)" }}>The dot product is identical.</strong> Same relative distance → same attention score, regardless of absolute position.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
