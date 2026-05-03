"use client";

import { useState } from "react";

// Shows the difference between:
// 1. Pre-tokenized BPE: split on spaces first, then run BPE on each word
// 2. SentencePiece BPE: run BPE on the raw character stream (spaces included)

interface PairCount {
  pair: string;
  count: number;
  crossWord: boolean;
}

function countPairs(tokens: string[]): PairCount[] {
  const counts = new Map<string, { count: number; crossWord: boolean }>();
  for (let i = 0; i < tokens.length - 1; i++) {
    const pair = `${tokens[i]} ${tokens[i + 1]}`;
    const existing = counts.get(pair) || { count: 0, crossWord: false };
    existing.count++;
    // Mark as cross-word if either token is ▁
    if (tokens[i] === "▁" || tokens[i + 1] === "▁") {
      existing.crossWord = true;
    }
    counts.set(pair, existing);
  }
  return Array.from(counts.entries())
    .map(([pair, { count, crossWord }]) => ({ pair, count, crossWord }))
    .sort((a, b) => b.count - a.count);
}

const preTokenizedSteps = [
  {
    label: "Raw text",
    display: "the cat sat on the mat",
  },
  {
    label: "Step 1: Pre-tokenize (split on spaces)",
    display: '["the", "cat", "sat", "on", "the", "mat"]',
    note: "BPE runs on each word SEPARATELY. It never sees pairs across words.",
  },
  {
    label: "Step 2: Split each word into characters",
    words: [
      { word: "the", chars: ["t", "h", "e"] },
      { word: "cat", chars: ["c", "a", "t"] },
      { word: "sat", chars: ["s", "a", "t"] },
      { word: "on", chars: ["o", "n"] },
      { word: "the", chars: ["t", "h", "e"] },
      { word: "mat", chars: ["m", "a", "t"] },
    ],
    note: "Each word is its own island. The pair (e, c) — end of 'the' and start of 'cat' — is NEVER counted.",
  },
  {
    label: "Step 3: Count pairs (within words only)",
    pairs: [
      { pair: "(t, h)", count: 2, note: "from 'the' ×2" },
      { pair: "(h, e)", count: 2, note: "from 'the' ×2" },
      { pair: "(a, t)", count: 3, note: "from 'cat', 'sat', 'mat'" },
      { pair: "(c, a)", count: 1, note: "" },
      { pair: "(s, a)", count: 1, note: "" },
      { pair: "(o, n)", count: 1, note: "" },
      { pair: "(m, a)", count: 1, note: "" },
    ],
    note: "Clean pairs — all within words. No noise. But we needed space-splitting rules to get here.",
  },
];

const sentencePieceSteps = [
  {
    label: "Raw text",
    display: "the cat sat on the mat",
  },
  {
    label: "Step 1: No pre-tokenization! Just add ▁ for spaces",
    chars: ["▁", "t", "h", "e", "▁", "c", "a", "t", "▁", "s", "a", "t", "▁", "o", "n", "▁", "t", "h", "e", "▁", "m", "a", "t"],
    note: "The entire text is one flat sequence. Spaces become ▁ and participate in pair counting.",
  },
  {
    label: "Step 2: Count ALL adjacent pairs",
    pairs: [
      { pair: "(a, t)", count: 3, cross: false, note: "within words — useful" },
      { pair: "(t, h)", count: 2, cross: false, note: "within words — useful" },
      { pair: "(h, e)", count: 2, cross: false, note: "within words — useful" },
      { pair: "(e, ▁)", count: 2, cross: true, note: "cross-word — noise" },
      { pair: "(t, ▁)", count: 2, cross: true, note: "cross-word — noise" },
      { pair: "(▁, t)", count: 2, cross: true, note: "cross-word — but useful! learns word-start" },
      { pair: "(▁, c)", count: 1, cross: true, note: "cross-word" },
      { pair: "(▁, s)", count: 1, cross: true, note: "cross-word" },
      { pair: "(▁, o)", count: 1, cross: true, note: "cross-word" },
      { pair: "(▁, m)", count: 1, cross: true, note: "cross-word" },
      { pair: "(o, n)", count: 1, cross: false, note: "within word" },
    ],
    note: "Yes, there are cross-word pairs like (e, ▁). But the useful pairs (a,t)=3 and (t,h)=2 still have the highest counts. The noise washes out with enough data.",
  },
  {
    label: "Step 3: First merge: (a, t) → at",
    result: ["▁", "t", "h", "e", "▁", "c", "at", "▁", "s", "at", "▁", "o", "n", "▁", "t", "h", "e", "▁", "m", "at"],
    note: "Same merge as pre-tokenized BPE would make! The cross-word noise didn't affect the outcome.",
  },
  {
    label: "After more merges: ▁t + h → ▁th → ▁the",
    result: ["▁the", "▁c", "at", "▁s", "at", "▁o", "n", "▁the", "▁m", "at"],
    note: "The ▁ merges WITH letters to form word-start tokens. This is how SentencePiece learns word boundaries from data — no rules needed.",
  },
];

export function SentencePieceDemo() {
  const [mode, setMode] = useState<"pretok" | "sp">("pretok");
  const [step, setStep] = useState(0);

  const steps = mode === "pretok" ? preTokenizedSteps : sentencePieceSteps;
  const current = steps[step];

  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        interactive comparison
      </p>
      <h4 className="mb-4 text-lg font-bold" style={{ fontFamily: "var(--font-serif)" }}>
        Pre-Tokenized BPE vs SentencePiece
      </h4>

      {/* Mode toggle */}
      <div className="mb-4 flex gap-2">
        <button
          onClick={() => { setMode("pretok"); setStep(0); }}
          className="rounded px-3 py-1.5 font-mono text-xs transition-colors"
          style={{
            backgroundColor: mode === "pretok" ? "var(--tag-bg)" : "transparent",
            color: mode === "pretok" ? "var(--primary)" : "var(--text-muted)",
            border: "1px solid var(--border)",
          }}
        >
          Traditional (pre-tokenize first)
        </button>
        <button
          onClick={() => { setMode("sp"); setStep(0); }}
          className="rounded px-3 py-1.5 font-mono text-xs transition-colors"
          style={{
            backgroundColor: mode === "sp" ? "var(--tag-bg)" : "transparent",
            color: mode === "sp" ? "var(--primary)" : "var(--text-muted)",
            border: "1px solid var(--border)",
          }}
        >
          SentencePiece (raw text)
        </button>
      </div>

      {/* Progress */}
      <div className="mb-4 flex gap-1">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className="h-1.5 flex-1 rounded-full transition-all"
            style={{
              backgroundColor: i <= step ? "var(--primary)" : "var(--border)",
              opacity: i === step ? 1 : i < step ? 0.5 : 0.3,
            }}
          />
        ))}
      </div>

      {/* Step content */}
      <div className="mb-4">
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: "var(--primary)" }}>
          {current.label}
        </p>

        {/* Simple display text */}
        {"display" in current && current.display && (
          <p className="mb-2 font-mono text-sm" style={{ color: "var(--text)" }}>
            {current.display}
          </p>
        )}

        {/* Character sequence */}
        {"chars" in current && current.chars && (
          <div className="mb-2 flex flex-wrap gap-0.5">
            {current.chars.map((c: string, i: number) => (
              <span
                key={i}
                className="rounded px-1.5 py-0.5 font-mono text-xs"
                style={{
                  backgroundColor: c === "▁" ? "rgba(139,175,122,0.15)" : "var(--tag-bg)",
                  color: c === "▁" ? "var(--sage)" : "var(--tag-text)",
                  border: "1px solid var(--border)",
                }}
              >
                {c}
              </span>
            ))}
          </div>
        )}

        {/* Token result */}
        {"result" in current && current.result && (
          <div className="mb-2 flex flex-wrap gap-0.5">
            {current.result.map((t: string, i: number) => (
              <span
                key={i}
                className="rounded px-1.5 py-0.5 font-mono text-xs"
                style={{
                  backgroundColor: t.startsWith("▁") && t.length > 1 ? "rgba(139,175,122,0.15)" : t.length > 1 ? "var(--tag-bg)" : "rgba(232,151,108,0.08)",
                  color: t.startsWith("▁") && t.length > 1 ? "var(--sage)" : t.length > 1 ? "var(--tag-text)" : "var(--text-muted)",
                  border: "1px solid var(--border)",
                }}
              >
                {t}
              </span>
            ))}
          </div>
        )}

        {/* Pre-tokenized word groups */}
        {"words" in current && current.words && (
          <div className="mb-2 space-y-1.5">
            {current.words.map((w: { word: string; chars: string[] }, i: number) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-10 shrink-0 text-right font-mono text-xs" style={{ color: "var(--text-muted)" }}>
                  {w.word}:
                </span>
                <div className="flex gap-0.5 rounded p-1" style={{ backgroundColor: "var(--bg)", border: "1px dashed var(--border)" }}>
                  {w.chars.map((c, j) => (
                    <span key={j} className="rounded px-1.5 py-0.5 font-mono text-xs" style={{ backgroundColor: "var(--tag-bg)", color: "var(--tag-text)" }}>
                      {c}
                    </span>
                  ))}
                </div>
                <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>←isolated</span>
              </div>
            ))}
          </div>
        )}

        {/* Pair counts */}
        {"pairs" in current && current.pairs && (
          <div className="mb-2 overflow-hidden rounded border" style={{ borderColor: "var(--border)" }}>
            <table className="w-full font-mono text-xs">
              <thead>
                <tr style={{ backgroundColor: "var(--bg-elevated)" }}>
                  <th className="px-3 py-1.5 text-left" style={{ color: "var(--text-muted)" }}>pair</th>
                  <th className="px-3 py-1.5 text-right" style={{ color: "var(--text-muted)" }}>count</th>
                  <th className="px-3 py-1.5 text-left" style={{ color: "var(--text-muted)" }}>note</th>
                </tr>
              </thead>
              <tbody>
                {current.pairs.map((p: { pair: string; count: number; cross?: boolean; note?: string }, i: number) => (
                  <tr
                    key={i}
                    style={{
                      borderTop: "1px solid var(--border)",
                      backgroundColor: p.cross ? "rgba(139,175,122,0.04)" : "transparent",
                    }}
                  >
                    <td className="px-3 py-1.5" style={{ color: p.cross ? "var(--text-muted)" : "var(--text-secondary)" }}>
                      {p.pair}
                    </td>
                    <td className="px-3 py-1.5 text-right" style={{ color: "var(--primary)" }}>
                      {p.count}
                    </td>
                    <td className="px-3 py-1.5" style={{ color: "var(--text-muted)", fontSize: "10px" }}>
                      {p.note || ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Note */}
        {"note" in current && current.note && (
          <div
            className="mt-3 rounded p-3"
            style={{ borderLeft: "3px solid var(--primary)", backgroundColor: "rgba(232,151,108,0.04)" }}
          >
            <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
              {current.note}
            </p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          className="rounded border px-3 py-1.5 font-mono text-xs transition-colors disabled:opacity-30"
          style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
        >
          ← prev
        </button>
        <button
          onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))}
          disabled={step === steps.length - 1}
          className="rounded px-3 py-1.5 font-mono text-xs text-white transition-colors disabled:opacity-30"
          style={{ backgroundColor: "var(--primary)" }}
        >
          next →
        </button>
        <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          {step + 1} / {steps.length}
        </span>
      </div>
    </div>
  );
}
