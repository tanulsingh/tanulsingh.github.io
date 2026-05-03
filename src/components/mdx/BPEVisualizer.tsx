"use client";

import { useState, useMemo, useEffect, useRef } from "react";

interface MergeStep {
  pair: string;
  pairA: string;
  pairB: string;
  count: number;
  tokens: string[];
  vocab: string[];
}

function computeBPESteps(text: string, maxSteps: number = 15): { initialTokens: string[]; steps: MergeStep[] } {
  let tokens = text.split("").map((c) => (c === " " ? "▁" : c));
  const initialTokens = [...tokens];
  const vocab = [...new Set(tokens)];
  const steps: MergeStep[] = [];

  for (let step = 0; step < maxSteps; step++) {
    const pairCounts = new Map<string, number>();
    for (let i = 0; i < tokens.length - 1; i++) {
      const pair = `${tokens[i]}|${tokens[i + 1]}`;
      pairCounts.set(pair, (pairCounts.get(pair) || 0) + 1);
    }

    if (pairCounts.size === 0) break;

    let bestPair = "";
    let bestCount = 0;
    for (const [pair, count] of pairCounts) {
      if (count > bestCount) {
        bestPair = pair;
        bestCount = count;
      }
    }

    if (bestCount < 1) break;

    const [a, b] = bestPair.split("|");
    const merged = a + b;
    const newTokens: string[] = [];
    let i = 0;
    while (i < tokens.length) {
      if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
        newTokens.push(merged);
        i += 2;
      } else {
        newTokens.push(tokens[i]);
        i++;
      }
    }

    vocab.push(merged);
    tokens = newTokens;

    steps.push({
      pair: `${a} + ${b} → ${merged}`,
      pairA: a,
      pairB: b,
      count: bestCount,
      tokens: [...tokens],
      vocab: [...vocab],
    });
  }

  return { initialTokens, steps };
}

function PairCountingAnimation({
  tokens,
  onComplete,
}: {
  tokens: string[];
  onComplete: () => void;
}) {
  const [scanIndex, setScanIndex] = useState(-1);
  const [pairCounts, setPairCounts] = useState<Map<string, number>>(new Map());
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [speed, setSpeed] = useState(350);

  function startScan() {
    setPairCounts(new Map());
    setScanIndex(0);
    setIsRunning(true);
    setIsDone(false);
  }

  useEffect(() => {
    if (!isRunning || scanIndex < 0) return;

    if (scanIndex >= tokens.length - 1) {
      setIsRunning(false);
      setIsDone(true);
      return;
    }

    const timer = setTimeout(() => {
      const pair = `${tokens[scanIndex]} ${tokens[scanIndex + 1]}`;
      setPairCounts((prev) => {
        const next = new Map(prev);
        next.set(pair, (next.get(pair) || 0) + 1);
        return next;
      });
      setScanIndex((s) => s + 1);
    }, speed);

    return () => clearTimeout(timer);
  }, [scanIndex, isRunning, tokens, speed]);

  const sortedPairs = [...pairCounts.entries()].sort((a, b) => b[1] - a[1]);

  return (
    <div>
      {/* Token row with scanning highlight */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          {isDone
            ? `scan complete — ${tokens.length - 1} pairs examined`
            : isRunning
            ? `scanning position ${scanIndex}...`
            : "click 'scan pairs' to watch the counting process"}
        </p>
        <div className="flex flex-wrap gap-0.5">
          {tokens.map((token, i) => {
            const isCurrentLeft = isRunning && i === scanIndex;
            const isCurrentRight = isRunning && i === scanIndex + 1;
            const isScanned = i < scanIndex;

            return (
              <span
                key={`${i}-${token}`}
                className="px-1.5 py-0.5 font-mono text-xs transition-all duration-100"
                style={{
                  backgroundColor:
                    isCurrentLeft || isCurrentRight
                      ? "rgba(232,151,108,0.25)"
                      : isScanned
                      ? "rgba(232,151,108,0.05)"
                      : "transparent",
                  color:
                    isCurrentLeft || isCurrentRight
                      ? "var(--primary)"
                      : isScanned
                      ? "var(--text-secondary)"
                      : "var(--text-muted)",
                  borderBottom:
                    isCurrentLeft || isCurrentRight
                      ? "2px solid var(--primary)"
                      : "2px solid transparent",
                }}
              >
                {token}
              </span>
            );
          })}
        </div>
      </div>

      {/* Live pair frequency table */}
      {(isRunning || isDone) && sortedPairs.length > 0 && (
        <div className="mb-4">
          <p className="mb-2 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            pair frequencies (updating live):
          </p>
          <div
            className="overflow-hidden rounded border"
            style={{ borderColor: "var(--border)" }}
          >
            <table className="w-full font-mono text-xs">
              <thead>
                <tr style={{ backgroundColor: "var(--bg-elevated)" }}>
                  <th className="px-3 py-1.5 text-left" style={{ color: "var(--text-muted)" }}>pair</th>
                  <th className="px-3 py-1.5 text-right" style={{ color: "var(--text-muted)" }}>count</th>
                  <th className="px-3 py-1.5 text-left" style={{ color: "var(--text-muted)" }}>frequency</th>
                </tr>
              </thead>
              <tbody>
                {sortedPairs.slice(0, 8).map(([pair, count]) => {
                  const maxCount = sortedPairs[0][1];
                  const barWidth = (count / maxCount) * 100;
                  const isTop = count === maxCount;

                  return (
                    <tr
                      key={pair}
                      style={{
                        borderTop: "1px solid var(--border)",
                        backgroundColor: isTop ? "rgba(232,151,108,0.06)" : "transparent",
                      }}
                    >
                      <td className="px-3 py-1.5" style={{ color: isTop ? "var(--primary)" : "var(--text-secondary)" }}>
                        ({pair})
                      </td>
                      <td className="px-3 py-1.5 text-right" style={{ color: isTop ? "var(--primary)" : "var(--text-secondary)" }}>
                        {count}
                      </td>
                      <td className="px-3 py-1.5">
                        <div className="flex items-center gap-2">
                          <div
                            className="h-2 rounded-sm transition-all duration-200"
                            style={{
                              width: `${barWidth}%`,
                              minWidth: "4px",
                              backgroundColor: isTop ? "var(--primary)" : "var(--border-hover)",
                            }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        {!isRunning && !isDone && (
          <button
            onClick={startScan}
            className="rounded px-3 py-1.5 font-mono text-xs text-white"
            style={{ backgroundColor: "var(--primary)" }}
          >
            scan pairs →
          </button>
        )}
        {isDone && (
          <>
            <button
              onClick={onComplete}
              className="rounded px-3 py-1.5 font-mono text-xs text-white"
              style={{ backgroundColor: "var(--primary)" }}
            >
              merge top pair →
            </button>
            <button
              onClick={startScan}
              className="rounded border px-3 py-1.5 font-mono text-xs"
              style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
            >
              rescan
            </button>
          </>
        )}
        {isRunning && (
          <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            scanning...
          </span>
        )}

        {/* Speed control */}
        <div className="ml-auto flex items-center gap-2">
          <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>speed:</span>
          {[
            { label: "slow", value: 500 },
            { label: "med", value: 300 },
            { label: "fast", value: 120 },
          ].map((s) => (
            <button
              key={s.label}
              onClick={() => setSpeed(s.value)}
              className="rounded px-2 py-0.5 font-mono text-xs transition-colors"
              style={{
                backgroundColor: speed === s.value ? "var(--tag-bg)" : "transparent",
                color: speed === s.value ? "var(--primary)" : "var(--text-muted)",
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export function BPEVisualizer() {
  const [input, setInput] = useState("the cat sat on the mat the cat");
  const [currentStep, setCurrentStep] = useState(-1);
  const [showScanner, setShowScanner] = useState(false);

  const { initialTokens, steps } = useMemo(() => computeBPESteps(input), [input]);

  const displayTokens = currentStep === -1 ? initialTokens : steps[currentStep]?.tokens || initialTokens;
  const currentMerge = currentStep >= 0 ? steps[currentStep] : null;

  function handleMerge() {
    setCurrentStep((s) => Math.min(s + 1, steps.length - 1));
    setShowScanner(false);
  }

  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        interactive
      </p>
      <h4
        className="mb-4 text-lg font-bold"
        style={{ fontFamily: "var(--font-serif)" }}
      >
        BPE Tokenizer — Step by Step
      </h4>

      {/* Input */}
      <div className="mb-4">
        <label className="mb-1 block font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          input text:
        </label>
        <input
          type="text"
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            setCurrentStep(-1);
            setShowScanner(false);
          }}
          className="w-full rounded border px-3 py-2 font-mono text-sm outline-none"
          style={{
            borderColor: "var(--border)",
            backgroundColor: "var(--bg)",
            color: "var(--text)",
          }}
        />
        <p className="mt-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          tip: use sentences with repeated words for best results, e.g. &quot;the cat sat on the mat the cat&quot;
        </p>
      </div>

      {/* Step info */}
      <div
        className="mb-4 rounded p-3"
        style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
      >
        <p className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          step {currentStep + 1} / {steps.length} · {displayTokens.length} tokens · vocab size: {currentStep === -1 ? new Set(initialTokens).size : currentMerge?.vocab.length}
        </p>
      </div>

      {/* Current merge result */}
      {currentMerge && (
        <div
          className="mb-4 rounded p-3"
          style={{
            borderLeft: "3px solid var(--primary)",
            backgroundColor: "rgba(232,151,108,0.06)",
          }}
        >
          <p className="font-mono text-sm" style={{ color: "var(--text-secondary)" }}>
            merge #{currentStep + 1}:{" "}
            <span style={{ color: "var(--primary)" }}>{currentMerge.pair}</span>{" "}
            (appeared {currentMerge.count}×)
          </p>
        </div>
      )}

      {/* Current tokens */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          current tokens:
        </p>
        <div className="flex flex-wrap gap-1">
          {displayTokens.map((token, i) => {
            const isNew =
              currentMerge &&
              token === currentMerge.pairA + currentMerge.pairB;
            return (
              <span
                key={`${i}-${token}`}
                className="rounded px-2 py-1 font-mono text-xs"
                style={{
                  backgroundColor: isNew ? "rgba(232,151,108,0.15)" : "var(--tag-bg)",
                  color: isNew ? "var(--primary)" : "var(--tag-text)",
                  border: isNew ? "1px solid var(--primary)" : "1px solid var(--border)",
                }}
              >
                {token}
              </span>
            );
          })}
        </div>
      </div>

      {/* Pair counting animation */}
      {showScanner && (
        <div
          className="mb-4 rounded border p-4"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--bg)" }}
        >
          <p className="mb-3 font-mono text-xs font-semibold" style={{ color: "var(--primary)" }}>
            Phase 1: Count adjacent pairs
          </p>
          <PairCountingAnimation
            tokens={displayTokens}
            onComplete={handleMerge}
          />
        </div>
      )}

      {/* Vocabulary */}
      {currentStep >= 0 && (
        <div className="mb-4">
          <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            vocabulary:
          </p>
          <div className="flex flex-wrap gap-1">
            {currentMerge?.vocab.map((v, i) => {
              const isNewest = i === (currentMerge?.vocab.length || 0) - 1;
              return (
                <span
                  key={`${i}-${v}`}
                  className="rounded px-1.5 py-0.5 font-mono text-xs"
                  style={{
                    backgroundColor: isNewest ? "rgba(232,151,108,0.12)" : "transparent",
                    color: isNewest ? "var(--primary)" : "var(--text-muted)",
                  }}
                >
                  {v}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => {
            setCurrentStep(-1);
            setShowScanner(false);
          }}
          disabled={currentStep === -1 && !showScanner}
          className="rounded border px-3 py-1.5 font-mono text-xs transition-colors disabled:opacity-30"
          style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
        >
          reset
        </button>

        {!showScanner && (
          <button
            onClick={() => setShowScanner(true)}
            disabled={currentStep >= steps.length - 1}
            className="rounded px-3 py-1.5 font-mono text-xs text-white transition-colors disabled:opacity-30"
            style={{ backgroundColor: "var(--primary)" }}
          >
            next iteration →
          </button>
        )}

        <button
          onClick={() => {
            for (let i = currentStep + 1; i < steps.length; i++) {
              setCurrentStep(i);
            }
            setCurrentStep(steps.length - 1);
            setShowScanner(false);
          }}
          disabled={currentStep >= steps.length - 1}
          className="rounded border px-3 py-1.5 font-mono text-xs transition-colors disabled:opacity-30"
          style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
        >
          run all
        </button>
      </div>
    </div>
  );
}
