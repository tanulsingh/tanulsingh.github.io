"use client";

import { motion, useInView } from "motion/react";
import { useRef, useState, useEffect } from "react";

interface Milestone {
  year: number;
  label: string;
  description: string;
  loss: number;
  accent?: boolean;
}

const milestones: Milestone[] = [
  { year: 2017, label: "The Spark", description: "Discovered ML in my 2nd year of Mechanical Engineering. Andrew Ng's course. Instantly hooked.", loss: 0.95 },
  { year: 2018, label: "The Hardest Year", description: "Lost my father. No earning member left, still in college. Tutored kids by day, studied ML at night.", loss: 0.90, accent: true },
  { year: 2019, label: "Found Kaggle", description: "Taught myself Python. Started writing Kaggle notebooks. Learned from experts I'd never meet.", loss: 0.75 },
  { year: 2020, label: "Notebooks GM", description: "Kaggle Notebooks Grandmaster. Competitions Expert. Secured first ML job at Javis.", loss: 0.55 },
  { year: 2021, label: "MLE at LevelAI", description: "Joined LevelAI as ML Engineer. Became Kaggle Competitions Master.", loss: 0.42 },
  { year: 2022, label: "Senior MLE", description: "Promoted to Senior MLE. Won Kaggle Days Championship — represented India in Barcelona.", loss: 0.30, accent: true },
  { year: 2023, label: "Patent & Paper", description: "US Patent granted for dynamic intent detection. Published GWNET research paper.", loss: 0.20 },
  { year: 2024, label: "Lead MLE", description: "Promoted to Lead ML Engineer at LevelAI. Built LLM autoscoring, RAG systems at scale.", loss: 0.14 },
  { year: 2025, label: "Apple", description: "Senior ML Engineer at Apple. Half research, half applied AI — the role I'd been training for.", loss: 0.08, accent: true },
];

const futurePoint = { year: 2028, label: "Research Dream", description: "The curve hasn't converged yet.", loss: 0.03 };

// --- Neural Network Visualization ---

const NN_CHART = { width: 900, height: 200, padX: 60, padY: 30 };

function NeuralNetworkViz({ isInView }: { isInView: boolean }) {
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [forwardProgress, setForwardProgress] = useState(-1);
  const [backpropProgress, setBackpropProgress] = useState(-1);

  useEffect(() => {
    if (!isInView) return;
    const timers: ReturnType<typeof setTimeout>[] = [];
    milestones.forEach((_, i) => {
      timers.push(setTimeout(() => setForwardProgress(i), 800 + i * 400));
    });
    timers.push(setTimeout(() => {
      let bp = milestones.length - 1;
      const interval = setInterval(() => {
        setBackpropProgress(bp);
        bp--;
        if (bp < 0) clearInterval(interval);
      }, 250);
    }, 800 + milestones.length * 400 + 500));
    return () => timers.forEach(clearTimeout);
  }, [isInView]);

  const nodeSpacing = (NN_CHART.width - NN_CHART.padX * 2) / (milestones.length - 1);
  const cy = NN_CHART.height / 2;

  return (
    <div className="relative mb-8">
      <svg viewBox={`0 0 ${NN_CHART.width} ${NN_CHART.height}`} className="w-full">
        {/* Connection lines between neurons */}
        {milestones.map((_, i) => {
          if (i === 0) return null;
          const x1 = NN_CHART.padX + (i - 1) * nodeSpacing;
          const x2 = NN_CHART.padX + i * nodeSpacing;
          const isForwardActive = forwardProgress >= i;
          const isBackpropActive = backpropProgress >= 0 && backpropProgress <= i;
          return (
            <g key={`conn-${i}`}>
              {/* Base line */}
              <line
                x1={x1} y1={cy} x2={x2} y2={cy}
                stroke="var(--border)"
                strokeWidth="2"
              />
              {/* Forward pass signal */}
              {isInView && isForwardActive && (
                <motion.line
                  x1={x1} y1={cy} x2={x2} y2={cy}
                  stroke="var(--primary)"
                  strokeWidth="2.5"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.3, delay: 0 }}
                />
              )}
              {/* Forward pass pulse (animated dot traveling along the line) */}
              {isInView && forwardProgress === i && (
                <motion.circle
                  r="4"
                  fill="var(--primary)"
                  initial={{ cx: x1, cy: cy, opacity: 1 }}
                  animate={{ cx: x2, cy: cy, opacity: 0 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                />
              )}
              {/* Backprop gradient signal */}
              {isInView && isBackpropActive && (
                <motion.line
                  x1={x2} y1={cy - 8} x2={x1} y2={cy - 8}
                  stroke="var(--sage)"
                  strokeWidth="1.5"
                  strokeDasharray="4 3"
                  opacity={0.7}
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.2 }}
                />
              )}
              {/* Backprop pulse */}
              {isInView && backpropProgress === i - 1 && (
                <motion.circle
                  r="3"
                  fill="var(--sage)"
                  initial={{ cx: x2, cy: cy - 8, opacity: 0.8 }}
                  animate={{ cx: x1, cy: cy - 8, opacity: 0 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                />
              )}
            </g>
          );
        })}

        {/* Neuron nodes */}
        {milestones.map((m, i) => {
          const cx_ = NN_CHART.padX + i * nodeSpacing;
          const isActive = forwardProgress >= i;
          const isBackpropDone = backpropProgress >= 0 && backpropProgress <= i;
          const nodeRadius = activeIdx === i ? 24 : 20;
          const strengthOpacity = isBackpropDone ? 1 : isActive ? 0.7 : 0.3;

          return (
            <g
              key={m.year}
              className="cursor-pointer"
              onMouseEnter={() => setActiveIdx(i)}
              onMouseLeave={() => setActiveIdx(null)}
              onClick={() => setActiveIdx(activeIdx === i ? null : i)}
            >
              {/* Glow ring for accent nodes */}
              {m.accent && isActive && (
                <motion.circle
                  cx={cx_} cy={cy} r={nodeRadius + 8}
                  fill="none"
                  stroke={m.accent ? "var(--accent)" : "var(--primary)"}
                  strokeWidth="1"
                  opacity={0.3}
                  initial={{ scale: 0 }}
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                />
              )}

              {/* Neuron circle */}
              {isInView && (
                <motion.circle
                  cx={cx_} cy={cy} r={nodeRadius}
                  fill={activeIdx === i ? "var(--primary)" : "var(--bg-surface)"}
                  stroke={m.accent ? "var(--accent)" : "var(--primary)"}
                  strokeWidth={isActive ? 2.5 : 1.5}
                  opacity={strengthOpacity}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.5 + i * 0.15, type: "spring", stiffness: 200 }}
                />
              )}

              {/* Year label inside neuron */}
              {isInView && (
                <motion.text
                  x={cx_} y={cy + 1}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={activeIdx === i ? "white" : "var(--text-muted)"}
                  fontSize="10"
                  fontFamily="var(--font-mono)"
                  fontWeight="600"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: strengthOpacity }}
                  transition={{ delay: 0.6 + i * 0.15 }}
                >
                  {`'${String(m.year).slice(2)}`}
                </motion.text>
              )}

              {/* Label below */}
              <text
                x={cx_} y={cy + 38}
                textAnchor="middle"
                fill={isActive ? "var(--text-secondary)" : "var(--text-muted)"}
                fontSize="10"
                fontFamily="var(--font-serif)"
              >
                {m.label}
              </text>
            </g>
          );
        })}

        {/* Forward / Backward labels */}
        <text x={NN_CHART.padX} y={cy - 30} fill="var(--primary)" fontSize="9" fontFamily="var(--font-mono)" opacity={forwardProgress >= 0 ? 0.8 : 0}>
          forward pass →
        </text>
        <text x={NN_CHART.width - NN_CHART.padX} y={cy - 30} textAnchor="end" fill="var(--sage)" fontSize="9" fontFamily="var(--font-mono)" opacity={backpropProgress >= 0 ? 0.8 : 0}>
          ← backprop (gradients)
        </text>
      </svg>

      {/* Tooltip */}
      {activeIdx !== null && (
        <div
          className="pointer-events-none absolute z-20 w-56 rounded border p-3 text-sm shadow-lg"
          style={{
            backgroundColor: "var(--bg-surface)",
            borderColor: "var(--border)",
            left: `${((NN_CHART.padX + activeIdx * nodeSpacing) / NN_CHART.width) * 100}%`,
            top: "70%",
            transform: "translateX(-50%)",
          }}
        >
          <p className="font-mono text-xs" style={{ color: "var(--primary)" }}>
            epoch {milestones[activeIdx].year} — {milestones[activeIdx].label}
          </p>
          <p className="mt-1" style={{ color: "var(--text-secondary)" }}>
            {milestones[activeIdx].description}
          </p>
          <p className="mt-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            loss: {milestones[activeIdx].loss.toFixed(2)}
          </p>
        </div>
      )}
    </div>
  );
}

// --- Loss Curve ---
// Standard chart: x-axis at bottom (time), y-axis at left (loss).
// High loss at top (small SVG y), low loss at bottom (large SVG y).
// The curve should visually go DOWN from top-left to bottom-right.

const CHART = { left: 60, right: 740, top: 40, bottom: 280, width: 800, height: 320 };

function toX(year: number): number {
  const minYear = 2015;
  const maxYear = 2029;
  return CHART.left + ((year - minYear) / (maxYear - minYear)) * (CHART.right - CHART.left);
}

// loss=1.0 → top of chart (SVG y = CHART.top)
// loss=0.0 → bottom of chart (SVG y = CHART.bottom)
function toY(loss: number): number {
  return CHART.bottom - loss * (CHART.bottom - CHART.top);
}

function buildCurvePath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return "";
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`;
  }
  return d;
}

function LossCurve({ isInView }: { isInView: boolean }) {
  const dataPoints = milestones.map((m) => ({ x: toX(m.year), y: toY(m.loss) }));
  const futureXY = { x: toX(futurePoint.year), y: toY(futurePoint.loss) };
  const curvePath = buildCurvePath(dataPoints);
  const lastPoint = dataPoints[dataPoints.length - 1];
  // Fill area between curve and bottom axis
  const fillPath = curvePath + ` L ${lastPoint.x} ${CHART.bottom} L ${dataPoints[0].x} ${CHART.bottom} Z`;

  return (
    <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} className="w-full">
      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map((v) => (
        <line
          key={v}
          x1={CHART.left} y1={toY(v)} x2={CHART.right} y2={toY(v)}
          stroke="var(--border)" strokeWidth="0.5" opacity="0.3"
        />
      ))}

      {/* X-axis at bottom */}
      <line x1={CHART.left} y1={CHART.bottom} x2={CHART.right} y2={CHART.bottom} stroke="var(--text-muted)" strokeWidth="1" />
      {/* Y-axis at left */}
      <line x1={CHART.left} y1={CHART.top} x2={CHART.left} y2={CHART.bottom} stroke="var(--text-muted)" strokeWidth="1" />

      {/* X-axis label */}
      <text x={CHART.right} y={CHART.bottom + 28} textAnchor="end" fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)">
        epochs (time)
      </text>

      {/* Y-axis labels */}
      <text x={CHART.left - 8} y={CHART.top + 4} textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
        1.0
      </text>
      <text x={CHART.left - 8} y={CHART.bottom + 4} textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
        0.0
      </text>
      <text x={15} y={(CHART.top + CHART.bottom) / 2} textAnchor="middle" fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)" transform={`rotate(-90, 15, ${(CHART.top + CHART.bottom) / 2})`}>
        loss
      </text>

      {/* Year tick marks on x-axis */}
      {milestones.map((m) => (
        <g key={m.year}>
          <line x1={toX(m.year)} y1={CHART.bottom} x2={toX(m.year)} y2={CHART.bottom + 6} stroke="var(--text-muted)" strokeWidth="1" />
          <text x={toX(m.year)} y={CHART.bottom + 18} textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
            {`'${String(m.year).slice(2)}`}
          </text>
        </g>
      ))}

      {/* Gradient fill under curve */}
      <defs>
        <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.12" />
          <stop offset="100%" stopColor="var(--primary)" stopOpacity="0.01" />
        </linearGradient>
      </defs>

      {isInView && (
        <motion.path
          d={fillPath}
          fill="url(#lossFill)"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1 }}
        />
      )}

      {/* Loss curve — high at top-left, decreasing to bottom-right */}
      {isInView && (
        <motion.path
          d={curvePath}
          stroke="var(--primary)"
          strokeWidth="2.5"
          strokeLinecap="round"
          fill="none"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 2.5, ease: "easeOut", delay: 0.3 }}
        />
      )}

      {/* Dashed future */}
      {isInView && (
        <motion.line
          x1={lastPoint.x} y1={lastPoint.y}
          x2={futureXY.x} y2={futureXY.y}
          stroke="var(--primary)"
          strokeWidth="2"
          strokeDasharray="6 4"
          opacity="0.4"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, delay: 2.8 }}
        />
      )}

      {/* Future point */}
      {isInView && (
        <motion.circle
          cx={futureXY.x} cy={futureXY.y} r="5"
          fill="none" stroke="var(--primary)" strokeWidth="1.5"
          strokeDasharray="3 2" opacity="0.5"
          initial={{ scale: 0 }} animate={{ scale: 1 }}
          transition={{ delay: 3.5 }}
        />
      )}
      <text x={futureXY.x} y={CHART.bottom + 18} textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)">
        ?
      </text>

      {/* Data points on the curve */}
      {dataPoints.map((pt, i) => (
        <g key={milestones[i].year}>
          {isInView && (
            <motion.circle
              cx={pt.x} cy={pt.y} r="5"
              fill="var(--bg-surface)"
              stroke={milestones[i].accent ? "var(--accent)" : "var(--primary)"}
              strokeWidth="2"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.3 + (i / milestones.length) * 2.5, type: "spring", stiffness: 300 }}
            />
          )}
        </g>
      ))}
    </svg>
  );
}

// --- Main Component ---

export function LearningCurve() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section className="py-24" ref={ref} id="training-loop">
      <div className="mx-auto max-w-5xl px-6">
        {/* Header */}
        <p className="mb-1 font-mono text-sm" style={{ color: "var(--text-muted)" }}>
          model.fit(life, epochs=&infin;, lr=persistence)
        </p>
        <h2
          className="mb-2 text-3xl font-bold tracking-tight md:text-4xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          My Training Curve
        </h2>
        <p className="mb-12" style={{ color: "var(--text-secondary)" }}>
          Each milestone is a neuron. Experiences flow forward. Lessons backpropagate.
          The loss is still decreasing.
        </p>

        {/* Neural Network */}
        <NeuralNetworkViz isInView={isInView} />

        {/* Legend */}
        <div className="mb-10 flex flex-wrap items-center gap-6 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          <span className="flex items-center gap-2">
            <span className="inline-block h-0.5 w-6" style={{ backgroundColor: "var(--primary)" }} />
            forward pass (life moving forward)
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block h-0.5 w-6 border-t border-dashed" style={{ borderColor: "var(--sage)" }} />
            backprop (lessons learned)
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-full border-2" style={{ borderColor: "var(--accent)" }} />
            turning points
          </span>
        </div>

        {/* Loss Curve */}
        <div className="relative">
          <p className="mb-3 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            training_loss.plot()
          </p>
          <LossCurve isInView={isInView} />
        </div>
      </div>
    </section>
  );
}
