"use client";

// Multi-head attention — static vertical flow with text explanation
// No animation, no slides — just a clear step-by-step with inline diagrams

export function MultiHeadVisualizer() {
  return (
    <div
      className="not-prose my-8 rounded border p-5"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <p className="mb-1 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
        visual walkthrough
      </p>
      <h4 className="mb-6 text-lg font-bold" style={{ fontFamily: "var(--font-serif)" }}>
        Multi-Head Attention — The Full Pipeline
      </h4>

      {/* Step 1 */}
      <div className="mb-6">
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: "var(--primary)" }}>
          1. Project input into Q, K, V
        </p>
        <p className="mb-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          Multiply the input X (3 tokens × 8 dims) by one big W_Q matrix (8×8) to get Q (3×8). Same for K and V. One matrix multiply per projection — not per head.
        </p>
        <div className="flex items-center gap-2 overflow-x-auto py-2 font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <span className="rounded px-2 py-3 text-center" style={{ backgroundColor: "rgba(232,151,108,0.08)", border: "1px solid var(--border)", minWidth: "60px" }}>
            X<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×8</span>
          </span>
          <span style={{ color: "var(--text-muted)" }}>×</span>
          <span className="rounded px-2 py-3 text-center" style={{ backgroundColor: "rgba(232,151,108,0.05)", border: "1px solid var(--border)", minWidth: "60px" }}>
            W_Q<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>8×8</span>
          </span>
          <span style={{ color: "var(--text-muted)" }}>=</span>
          <span className="rounded px-2 py-3 text-center" style={{ backgroundColor: "rgba(232,151,108,0.15)", border: "1px solid var(--primary)", minWidth: "60px" }}>
            Q<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×8</span>
          </span>
          <span className="ml-4" style={{ color: "var(--text-muted)", fontSize: "10px" }}>
            (same for K and V)
          </span>
        </div>
      </div>

      {/* Step 2 */}
      <div className="mb-6">
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: "var(--primary)" }}>
          2. Split into h=2 heads (reshape, no computation)
        </p>
        <p className="mb-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          Reshape Q from (3×8) into 2 heads of (3×4). Dimensions 0-3 go to Head 1, dimensions 4-7 to Head 2. Same numbers, different grouping. Same split for K and V.
        </p>
        <div className="flex items-center gap-2 overflow-x-auto py-2 font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <span className="rounded px-2 py-3 text-center" style={{ backgroundColor: "rgba(232,151,108,0.15)", border: "1px solid var(--primary)", minWidth: "60px" }}>
            Q<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×8</span>
          </span>
          <span style={{ color: "var(--text-muted)" }}>→ reshape →</span>
          <span className="rounded px-2 py-3 text-center" style={{ backgroundColor: "rgba(232,151,108,0.2)", border: "1px dashed var(--primary)", minWidth: "70px" }}>
            Q₁<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×4 (dims 0-3)</span>
          </span>
          <span className="rounded px-2 py-3 text-center" style={{ backgroundColor: "rgba(212,168,67,0.2)", border: "1px dashed var(--accent)", minWidth: "70px" }}>
            Q₂<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×4 (dims 4-7)</span>
          </span>
        </div>
      </div>

      {/* Step 3 */}
      <div className="mb-6">
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: "var(--primary)" }}>
          3. Each head computes attention independently (in parallel)
        </p>
        <p className="mb-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          Each head runs the full attention computation on its own subspace. Different subspace → different attention patterns learned. These run simultaneously on the GPU.
        </p>
        <div className="space-y-2">
          <div className="flex items-center gap-2 overflow-x-auto rounded p-2 font-mono text-xs" style={{ border: "1px dashed var(--primary)", backgroundColor: "rgba(232,151,108,0.03)" }}>
            <span style={{ color: "var(--primary)", minWidth: "45px" }}>Head 1:</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(232,151,108,0.12)", fontSize: "10px" }}>Q₁</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(232,151,108,0.08)", fontSize: "10px" }}>K₁ᵀ</span>
            <span style={{ color: "var(--text-muted)" }}>→ softmax →</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(232,151,108,0.15)", fontSize: "10px" }}>Attn₁ (3×3)</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(232,151,108,0.08)", fontSize: "10px" }}>V₁</span>
            <span style={{ color: "var(--text-muted)" }}>=</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(232,151,108,0.2)", fontSize: "10px" }}>Out₁ (3×4)</span>
          </div>
          <div className="flex items-center gap-2 overflow-x-auto rounded p-2 font-mono text-xs" style={{ border: "1px dashed var(--accent)", backgroundColor: "rgba(212,168,67,0.03)" }}>
            <span style={{ color: "var(--accent)", minWidth: "45px" }}>Head 2:</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(212,168,67,0.12)", fontSize: "10px" }}>Q₂</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(212,168,67,0.08)", fontSize: "10px" }}>K₂ᵀ</span>
            <span style={{ color: "var(--text-muted)" }}>→ softmax →</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(212,168,67,0.15)", fontSize: "10px" }}>Attn₂ (3×3)</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(212,168,67,0.08)", fontSize: "10px" }}>V₂</span>
            <span style={{ color: "var(--text-muted)" }}>=</span>
            <span className="rounded px-1.5 py-1 text-center" style={{ backgroundColor: "rgba(212,168,67,0.2)", fontSize: "10px" }}>Out₂ (3×4)</span>
          </div>
        </div>
      </div>

      {/* Step 4 */}
      <div className="mb-6">
        <p className="mb-2 font-mono text-xs font-semibold" style={{ color: "var(--primary)" }}>
          4. Concatenate heads + output projection
        </p>
        <p className="mb-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          Stack head outputs side by side, then multiply by W_O to let heads share information. Without W_O, heads would be forever independent — it&apos;s the only cross-head interaction.
        </p>
        <div className="flex items-center gap-2 overflow-x-auto py-2 font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <span className="rounded px-1.5 py-3 text-center" style={{ backgroundColor: "rgba(232,151,108,0.2)", border: "1px solid var(--border)", minWidth: "50px" }}>
            Out₁<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×4</span>
          </span>
          <span style={{ color: "var(--text-muted)" }}>+</span>
          <span className="rounded px-1.5 py-3 text-center" style={{ backgroundColor: "rgba(212,168,67,0.2)", border: "1px solid var(--border)", minWidth: "50px" }}>
            Out₂<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×4</span>
          </span>
          <span style={{ color: "var(--text-muted)" }}>→ concat →</span>
          <span className="rounded px-1.5 py-3 text-center" style={{ border: "1px solid var(--border)", minWidth: "60px" }}>
            <span style={{ color: "var(--primary)" }}>▌</span><span style={{ color: "var(--accent)" }}>▌</span>
            <br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×8</span>
          </span>
          <span style={{ color: "var(--text-muted)" }}>× W_O →</span>
          <span className="rounded px-1.5 py-3 text-center" style={{ backgroundColor: "rgba(139,175,122,0.15)", border: "1px solid var(--sage)", minWidth: "60px" }}>
            Output<br/><span style={{ fontSize: "8px", color: "var(--text-muted)" }}>3×8</span>
          </span>
        </div>
      </div>

      {/* Summary */}
      <div className="rounded p-3" style={{ borderLeft: "3px solid var(--primary)", backgroundColor: "rgba(232,151,108,0.04)" }}>
        <p className="font-mono text-xs" style={{ color: "var(--text-secondary)" }}>
          <strong style={{ color: "var(--primary)" }}>Full pipeline:</strong> X → ×W_Q,K,V → split heads → parallel attention → concat → ×W_O → output.
          Total cost = same as single-head. You get h different attention perspectives for free.
        </p>
      </div>
    </div>
  );
}
