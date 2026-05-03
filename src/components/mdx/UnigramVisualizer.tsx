"use client";

import { useState } from "react";

// Pre-computed walkthrough of Unigram on a realistic (simulated) corpus.
// Shows what each step looks like with real-scale data.

interface Step {
  title: string;
  description: string;
  vocab?: { token: string; count?: number; prob?: number; isChar?: boolean; usage?: number }[];
  segmentations?: { word: string; segments: string[]; formula?: string; likelihood?: string }[];
  pruned?: string[];
  highlight?: string;
}

const steps: Step[] = [
  {
    title: "Corpus",
    description:
      "Imagine we have a training corpus with thousands of sentences. For this walkthrough, let's focus on how the algorithm handles these words that appear frequently:",
    segmentations: [
      { word: "tokenization", segments: ["t","o","k","e","n","i","z","a","t","i","o","n"] },
      { word: "token", segments: ["t","o","k","e","n"] },
      { word: "tokens", segments: ["t","o","k","e","n","s"] },
      { word: "running", segments: ["r","u","n","n","i","n","g"] },
      { word: "runner", segments: ["r","u","n","n","e","r"] },
      { word: "unhappy", segments: ["u","n","h","a","p","p","y"] },
      { word: "happy", segments: ["h","a","p","p","y"] },
    ],
    highlight: "At this point every word is split into individual characters. That's our starting point.",
  },
  {
    title: "Step 1 — Build Seed Vocabulary",
    description:
      "Extract ALL substrings (up to length 8) from every word in the corpus. Count how often each substring appears across all words.",
    vocab: [
      { token: "t", count: 14200, isChar: true },
      { token: "o", count: 9800, isChar: true },
      { token: "n", count: 11500, isChar: true },
      { token: "e", count: 8900, isChar: true },
      { token: "i", count: 7200, isChar: true },
      { token: "token", count: 3200 },
      { token: "toke", count: 3200 },
      { token: "oken", count: 3200 },
      { token: "tok", count: 3500 },
      { token: "oke", count: 3300 },
      { token: "ken", count: 3400 },
      { token: "to", count: 8100 },
      { token: "ok", count: 4200 },
      { token: "ke", count: 3900 },
      { token: "en", count: 6700 },
      { token: "tion", count: 4800 },
      { token: "ation", count: 3100 },
      { token: "ization", count: 1800 },
      { token: "iz", count: 2100 },
      { token: "za", count: 1900 },
      { token: "run", count: 2800 },
      { token: "nn", count: 2500 },
      { token: "ing", count: 5200 },
      { token: "ning", count: 2400 },
      { token: "un", count: 4100 },
      { token: "happy", count: 1900 },
      { token: "happ", count: 1900 },
      { token: "app", count: 2200 },
      { token: "pp", count: 2100 },
      { token: "py", count: 1900 },
    ],
    highlight:
      "This gives us ~150,000 candidate tokens. Notice: 'token' and 'tok' and 'to' all exist. Massive redundancy — that's intentional.",
  },
  {
    title: "Step 2a — Initial Probabilities (Raw Counts)",
    description:
      "Divide each substring's count by the total count of ALL substrings. This gives a rough initial probability — NOT based on any segmentation yet.",
    vocab: [
      { token: "t", count: 14200, prob: 0.0284, isChar: true },
      { token: "n", count: 11500, prob: 0.0230, isChar: true },
      { token: "o", count: 9800, prob: 0.0196, isChar: true },
      { token: "e", count: 8900, prob: 0.0178, isChar: true },
      { token: "to", count: 8100, prob: 0.0162 },
      { token: "i", count: 7200, prob: 0.0144, isChar: true },
      { token: "en", count: 6700, prob: 0.0134 },
      { token: "ing", count: 5200, prob: 0.0104 },
      { token: "tion", count: 4800, prob: 0.0096 },
      { token: "ok", count: 4200, prob: 0.0084 },
      { token: "un", count: 4100, prob: 0.0082 },
      { token: "ke", count: 3900, prob: 0.0078 },
      { token: "tok", count: 3500, prob: 0.0070 },
      { token: "token", count: 3200, prob: 0.0064 },
      { token: "ation", count: 3100, prob: 0.0062 },
      { token: "run", count: 2800, prob: 0.0056 },
    ],
    highlight:
      "P(token) = 3200 / 500000 = 0.0064. These are just raw frequencies — the EM step will refine them.",
  },
  {
    title: "Step 2b — EM: Segment Each Word",
    description:
      "Using these initial probabilities, find the BEST segmentation for each word. The algorithm tries every possible split and picks the one with highest P(x₁) × P(x₂) × ...",
    segmentations: [
      {
        word: "tokenization",
        segments: ["token", "ization"],
        formula: "P(token)×P(ization) = 0.0064 × 0.0036",
        likelihood: "2.3e-5",
      },
      {
        word: "token",
        segments: ["token"],
        formula: "P(token) = 0.0064",
        likelihood: "6.4e-3",
      },
      {
        word: "tokens",
        segments: ["token", "s"],
        formula: "P(token)×P(s) = 0.0064 × 0.0150",
        likelihood: "9.6e-5",
      },
      {
        word: "running",
        segments: ["run", "ning"],
        formula: "P(run)×P(ning) = 0.0056 × 0.0048",
        likelihood: "2.7e-5",
      },
      {
        word: "runner",
        segments: ["run", "n", "e", "r"],
        formula: "P(run)×P(n)×P(e)×P(r) = ...",
        likelihood: "1.1e-7",
      },
      {
        word: "unhappy",
        segments: ["un", "happy"],
        formula: "P(un)×P(happy) = 0.0082 × 0.0038",
        likelihood: "3.1e-5",
      },
      {
        word: "happy",
        segments: ["happy"],
        formula: "P(happy) = 0.0038",
        likelihood: "3.8e-3",
      },
    ],
    highlight:
      "Notice: 'tokenization' splits as [token·ization] not [t·o·k·e·n·i·z·a·t·i·o·n] because the product of two high-probability tokens beats twelve low-probability characters.",
  },
  {
    title: "Step 2b — Why 'token' Beats 't·o·k·e·n'",
    description:
      "Let's compare all possible segmentations of 'token' to see why the algorithm picks the whole word:",
    segmentations: [
      {
        word: "token (option 1)",
        segments: ["token"],
        formula: "P(token) = 0.0064",
        likelihood: "6.4e-3  ← WINNER",
      },
      {
        word: "token (option 2)",
        segments: ["tok", "en"],
        formula: "P(tok)×P(en) = 0.0070 × 0.0134",
        likelihood: "9.4e-5",
      },
      {
        word: "token (option 3)",
        segments: ["to", "ken"],
        formula: "P(to)×P(ken) = 0.0162 × 0.0068",
        likelihood: "1.1e-4",
      },
      {
        word: "token (option 4)",
        segments: ["t", "o", "k", "e", "n"],
        formula: "P(t)×P(o)×P(k)×P(e)×P(n) = 0.028×0.020×0.008×0.018×0.023",
        likelihood: "1.9e-9",
      },
    ],
    highlight:
      "6.4e-3 >> 1.1e-4 >> 9.4e-5 >> 1.9e-9. The whole word wins by orders of magnitude. Each split multiplies small numbers, making the product tiny.",
  },
  {
    title: "Step 2c — EM: Recount & Update Probabilities",
    description:
      "Now count how often each token was USED in those best segmentations. Tokens that weren't used in any segmentation get very low probabilities.",
    vocab: [
      { token: "token", usage: 3200, prob: 0.0520 },
      { token: "ing", usage: 2800, prob: 0.0455 },
      { token: "happy", usage: 1900, prob: 0.0309 },
      { token: "t", usage: 1800, prob: 0.0293, isChar: true },
      { token: "run", usage: 1600, prob: 0.0260 },
      { token: "tion", usage: 1400, prob: 0.0228 },
      { token: "un", usage: 1200, prob: 0.0195 },
      { token: "n", usage: 1100, prob: 0.0179, isChar: true },
      { token: "e", usage: 900, prob: 0.0146, isChar: true },
      { token: "ization", usage: 800, prob: 0.0130 },
      { token: "toke", usage: 0, prob: 0.0001 },
      { token: "oken", usage: 0, prob: 0.0001 },
      { token: "oke", usage: 0, prob: 0.0001 },
      { token: "ok", usage: 0, prob: 0.0001 },
      { token: "ke", usage: 0, prob: 0.0001 },
    ],
    highlight:
      "Key insight: 'toke', 'oken', 'oke', 'ok', 'ke' have usage = 0! They exist in the vocabulary but were never chosen in any best segmentation. These are the tokens we'll prune.",
  },
  {
    title: "Steps 3-4 — Evaluate & Prune",
    description:
      "For each token, ask: 'if I remove this, how do segmentations change?' Tokens with zero or near-zero usage barely affect the corpus — safe to remove.",
    vocab: [
      { token: "toke", usage: 0, prob: 0.0001 },
      { token: "oken", usage: 0, prob: 0.0001 },
      { token: "oke", usage: 0, prob: 0.0001 },
      { token: "ok", usage: 0, prob: 0.0001 },
      { token: "ke", usage: 0, prob: 0.0001 },
      { token: "za", usage: 0, prob: 0.0001 },
      { token: "iz", usage: 0, prob: 0.0001 },
      { token: "nn", usage: 0, prob: 0.0001 },
    ],
    pruned: ["toke", "oken", "oke", "ok", "ke", "za", "iz", "nn", "happ", "app"],
    highlight:
      "We prune ~15% of the vocabulary. These were all redundant — their subparts handle the same text. Single characters (t, o, k, ...) are NEVER pruned — they're the safety net.",
  },
  {
    title: "Step 5 — Repeat",
    description:
      "With the pruned vocabulary, run EM again. Now some segmentations change because tokens they relied on are gone. For example, if we removed 'ning':",
    segmentations: [
      {
        word: "running (before)",
        segments: ["run", "ning"],
        formula: "P(run)×P(ning)",
        likelihood: "2.7e-5",
      },
      {
        word: "running (after prune)",
        segments: ["run", "n", "ing"],
        formula: "P(run)×P(n)×P(ing)",
        likelihood: "3.4e-6",
      },
      {
        word: "tokenization",
        segments: ["token", "ization"],
        formula: "P(token)×P(ization) — unchanged, both tokens survived",
        likelihood: "2.3e-5",
      },
    ],
    highlight:
      "The likelihood drops slightly (2.7e-5 → 3.4e-6) but the text is still segmentable. We keep pruning until the vocabulary reaches our target size (e.g., 32,000 tokens).",
  },
  {
    title: "Final Result",
    description:
      "After many rounds of EM + pruning, the vocabulary settles into meaningful subword units. Common words stay whole. Rare words decompose into reusable parts.",
    segmentations: [
      { word: "tokenization", segments: ["token", "ization"], likelihood: "common word + common suffix" },
      { word: "token", segments: ["token"], likelihood: "frequent — stays as one token" },
      { word: "tokens", segments: ["token", "s"], likelihood: "word + plural marker" },
      { word: "running", segments: ["run", "ing"], likelihood: "stem + suffix" },
      { word: "runner", segments: ["run", "er"], likelihood: "stem + agent suffix" },
      { word: "unhappy", segments: ["un", "happy"], likelihood: "prefix + word" },
      { word: "unhappiness", segments: ["un", "happy", "ness"], likelihood: "prefix + word + suffix" },
    ],
    highlight:
      "The algorithm learned morphology (un-, -ing, -tion, -ness) without any linguistic rules — purely from statistics. This is the power of Unigram.",
  },
];

export function UnigramVisualizer() {
  const [currentStep, setCurrentStep] = useState(0);
  const step = steps[currentStep];

  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        interactive walkthrough
      </p>
      <h4 className="mb-4 text-lg font-bold" style={{ fontFamily: "var(--font-serif)" }}>
        Unigram Tokenizer — How It Works
      </h4>

      {/* Progress bar */}
      <div className="mb-4 flex gap-1">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrentStep(i)}
            className="h-1.5 flex-1 rounded-full transition-all"
            style={{
              backgroundColor: i <= currentStep ? "var(--primary)" : "var(--border)",
              opacity: i === currentStep ? 1 : i < currentStep ? 0.5 : 0.3,
            }}
          />
        ))}
      </div>

      {/* Step header */}
      <div className="mb-4">
        <p className="mb-1 font-mono text-xs" style={{ color: "var(--primary)" }}>
          {currentStep + 1} / {steps.length}
        </p>
        <h5 className="mb-2 text-base font-bold">{step.title}</h5>
        <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          {step.description}
        </p>
      </div>

      {/* Vocabulary table */}
      {step.vocab && (
        <div
          className="mb-4 max-h-52 overflow-y-auto rounded border"
          style={{ borderColor: "var(--border)" }}
        >
          <table className="w-full font-mono text-xs">
            <thead>
              <tr style={{ backgroundColor: "var(--bg-elevated)", position: "sticky", top: 0 }}>
                <th className="px-3 py-1.5 text-left" style={{ color: "var(--text-muted)" }}>token</th>
                {step.vocab[0].count !== undefined && (
                  <th className="px-3 py-1.5 text-right" style={{ color: "var(--text-muted)" }}>
                    {step.vocab[0].usage !== undefined ? "usage" : "count"}
                  </th>
                )}
                {step.vocab[0].prob !== undefined && (
                  <th className="px-3 py-1.5 text-right" style={{ color: "var(--text-muted)" }}>probability</th>
                )}
                <th className="px-3 py-1.5 text-left" style={{ color: "var(--text-muted)" }}>note</th>
              </tr>
            </thead>
            <tbody>
              {step.vocab.map((token) => {
                const isPruned = step.pruned?.includes(token.token);
                return (
                  <tr
                    key={token.token}
                    style={{
                      borderTop: "1px solid var(--border)",
                      backgroundColor: isPruned
                        ? "rgba(139,175,122,0.06)"
                        : token.usage === 0
                        ? "rgba(232,151,108,0.04)"
                        : "transparent",
                    }}
                  >
                    <td
                      className="px-3 py-1.5"
                      style={{
                        color: token.isChar ? "var(--sage)" : isPruned ? "var(--text-muted)" : "var(--text-secondary)",
                        textDecoration: isPruned ? "line-through" : "none",
                      }}
                    >
                      <span className="font-semibold">{token.token}</span>
                      {token.isChar && " ⊘"}
                    </td>
                    {(token.count !== undefined || token.usage !== undefined) && (
                      <td className="px-3 py-1.5 text-right" style={{ color: "var(--text-muted)" }}>
                        {token.usage !== undefined ? token.usage.toLocaleString() : token.count?.toLocaleString()}
                      </td>
                    )}
                    {token.prob !== undefined && (
                      <td className="px-3 py-1.5 text-right" style={{ color: "var(--primary)" }}>
                        {token.prob.toFixed(4)}
                      </td>
                    )}
                    <td className="px-3 py-1.5" style={{ color: "var(--text-muted)", fontSize: "10px" }}>
                      {token.isChar && "protected — never removed"}
                      {token.usage === 0 && !token.isChar && "⚠ unused — prune candidate"}
                      {isPruned && "removed ✕"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Segmentations */}
      {step.segmentations && (
        <div className="mb-4 space-y-2">
          {step.segmentations.map((seg, i) => (
            <div
              key={i}
              className="rounded p-2.5"
              style={{ backgroundColor: "var(--bg)", border: "1px solid var(--border)" }}
            >
              <div className="flex items-center gap-2 flex-wrap">
                <span className="shrink-0 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
                  {seg.word}→
                </span>
                <div className="flex gap-0.5">
                  {seg.segments.map((s, j) => (
                    <span
                      key={j}
                      className="rounded px-1.5 py-0.5 font-mono text-xs"
                      style={{
                        backgroundColor: s.length > 1 ? "var(--tag-bg)" : "rgba(139,175,122,0.1)",
                        color: s.length > 1 ? "var(--tag-text)" : "var(--sage)",
                        border: "1px solid var(--border)",
                      }}
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              {seg.formula && (
                <p className="mt-1 font-mono" style={{ color: "var(--text-muted)", fontSize: "10px" }}>
                  {seg.formula} = {seg.likelihood}
                </p>
              )}
              {!seg.formula && seg.likelihood && (
                <p className="mt-1 font-mono" style={{ color: "var(--text-muted)", fontSize: "10px" }}>
                  {seg.likelihood}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pruned tokens */}
      {step.pruned && (
        <div
          className="mb-4 rounded p-3"
          style={{ borderLeft: "3px solid var(--sage)", backgroundColor: "rgba(139,175,122,0.06)" }}
        >
          <p className="mb-1 font-mono text-xs" style={{ color: "var(--sage)" }}>
            pruned {step.pruned.length} tokens:
          </p>
          <div className="flex flex-wrap gap-1">
            {step.pruned.map((t) => (
              <span key={t} className="rounded px-1.5 py-0.5 font-mono text-xs line-through" style={{ color: "var(--text-muted)" }}>
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Highlight/insight */}
      {step.highlight && (
        <div
          className="mb-4 rounded p-3"
          style={{ borderLeft: "3px solid var(--primary)", backgroundColor: "rgba(232,151,108,0.04)" }}
        >
          <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
            {step.highlight}
          </p>
        </div>
      )}

      {/* Navigation */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
          disabled={currentStep === 0}
          className="rounded border px-3 py-1.5 font-mono text-xs transition-colors disabled:opacity-30"
          style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
        >
          ← prev
        </button>
        <button
          onClick={() => setCurrentStep((s) => Math.min(steps.length - 1, s + 1))}
          disabled={currentStep === steps.length - 1}
          className="rounded px-3 py-1.5 font-mono text-xs text-white transition-colors disabled:opacity-30"
          style={{ backgroundColor: "var(--primary)" }}
        >
          next →
        </button>
        <span className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
          {step.title}
        </span>
      </div>
    </div>
  );
}
