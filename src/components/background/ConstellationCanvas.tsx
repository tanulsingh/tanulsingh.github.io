"use client";

import { useEffect, useRef } from "react";

// Hybrid: nodes arranged in loose vertical layers (suggesting transformer depth)
// with sparse connections between layers. Forward pulses flow left-to-right,
// backward gradient pulses flow right-to-left. Layers are labeled faintly.
// Not a rigid diagram — organic, breathing, recognizable.

interface NetNode {
  x: number;
  y: number;
  layer: number;
  radius: number;
  baseAlpha: number;
  activation: number;
}

interface Edge {
  from: number;
  to: number;
  forwardProgress: number;
  backwardProgress: number;
}

interface Signal {
  path: number[];
  progress: number;
  speed: number;
  isBackward: boolean;
  active: boolean;
}

export function ConstellationCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const stateRef = useRef<{
    nodes: NetNode[];
    edges: Edge[];
    signals: Signal[];
    lastSignalTime: number;
    layerLabels: { x: number; label: string }[];
    width: number;
    height: number;
  }>({ nodes: [], edges: [], signals: [], lastSignalTime: 0, layerLabels: [], width: 0, height: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function getColors() {
      const isLight = document.documentElement.classList.contains("light");
      return {
        node: isLight ? "196,122,82" : "232,151,108",
        edge: isLight ? "196,122,82" : "232,151,108",
        forward: isLight ? "196,122,82" : "232,151,108",
        backward: isLight ? "107,143,90" : "139,175,122",
        label: isLight ? "140,133,123" : "90,84,76",
      };
    }

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      canvas!.style.width = w + "px";
      canvas!.style.height = h + "px";
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      stateRef.current.width = w;
      stateRef.current.height = h;
    }

    function buildNetwork() {
      const w = stateRef.current.width;
      const h = stateRef.current.height;
      const isMobile = w < 768;

      // Layers loosely named after transformer components
      const layerDefs = isMobile
        ? [
            { name: "Embed", count: 4 },
            { name: "Q K V", count: 6 },
            { name: "Attn", count: 8 },
            { name: "FFN", count: 6 },
            { name: "Out", count: 4 },
          ]
        : [
            { name: "Embed", count: 5 },
            { name: "Q", count: 4 },
            { name: "K", count: 4 },
            { name: "V", count: 4 },
            { name: "Attn", count: 8 },
            { name: "Proj", count: 5 },
            { name: "FFN₁", count: 7 },
            { name: "FFN₂", count: 5 },
            { name: "Norm", count: 4 },
            { name: "Out", count: 5 },
          ];

      const layerCount = layerDefs.length;
      const nodes: NetNode[] = [];
      const edges: Edge[] = [];
      const layerLabels: { x: number; label: string }[] = [];

      const padX = w * 0.06;
      const padY = h * 0.1;
      const usableW = w - padX * 2;
      const usableH = h - padY * 2;

      for (let l = 0; l < layerCount; l++) {
        const def = layerDefs[l];
        const baseX = padX + (l / (layerCount - 1)) * usableW;

        layerLabels.push({ x: baseX, label: def.name });

        for (let n = 0; n < def.count; n++) {
          const ySpread = usableH * 0.8;
          const yOffset = (usableH - ySpread) / 2;
          const y = padY + yOffset + (def.count > 1 ? (n / (def.count - 1)) * ySpread : ySpread / 2);

          // Organic jitter
          const jitterX = (Math.random() - 0.5) * usableW * 0.025;
          const jitterY = (Math.random() - 0.5) * ySpread * 0.06;

          nodes.push({
            x: baseX + jitterX,
            y: y + jitterY,
            layer: l,
            radius: isMobile ? 2.5 : 3,
            baseAlpha: 0.12 + Math.random() * 0.08,
            activation: 0,
          });
        }
      }

      // Sparse connections to nearest neighbors in next layer
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const nextLayerNodes = nodes
          .map((n, idx) => ({ n, idx }))
          .filter((x) => x.n.layer === node.layer + 1);

        if (nextLayerNodes.length === 0) continue;

        // Sort by vertical proximity, connect to 2-3 nearest
        const sorted = [...nextLayerNodes].sort(
          (a, b) => Math.abs(a.n.y - node.y) - Math.abs(b.n.y - node.y)
        );
        const connectCount = 2 + Math.floor(Math.random() * 2);

        for (let j = 0; j < Math.min(connectCount, sorted.length); j++) {
          // Skip some connections for sparsity
          if (Math.random() > 0.7) continue;
          edges.push({
            from: i,
            to: sorted[j].idx,
            forwardProgress: 0,
            backwardProgress: 0,
          });
        }
      }

      stateRef.current.nodes = nodes;
      stateRef.current.edges = edges;
      stateRef.current.signals = [];
      stateRef.current.layerLabels = layerLabels;
      stateRef.current.lastSignalTime = performance.now();
    }

    function findPath(nodes: NetNode[], edges: Edge[]): number[] {
      const inputNodes = nodes.map((n, i) => ({ n, i })).filter((x) => x.n.layer === 0);
      if (inputNodes.length === 0) return [];
      let current = inputNodes[Math.floor(Math.random() * inputNodes.length)].i;
      const path = [current];

      const maxLayer = Math.max(...nodes.map((n) => n.layer));
      for (let l = 0; l < maxLayer; l++) {
        const outEdges = edges.filter((e) => e.from === current);
        if (outEdges.length === 0) break;
        const edge = outEdges[Math.floor(Math.random() * outEdges.length)];
        current = edge.to;
        path.push(current);
      }
      return path;
    }

    function spawnSignal() {
      const { nodes, edges } = stateRef.current;
      const path = findPath(nodes, edges);
      if (path.length < 2) return;
      stateRef.current.signals.push({
        path,
        progress: 0,
        speed: 0.006 + Math.random() * 0.004,
        isBackward: false,
        active: true,
      });
    }

    function draw() {
      const { nodes, edges, signals, layerLabels, width: w, height: h } = stateRef.current;
      const colors = getColors();
      const now = performance.now();

      ctx!.clearRect(0, 0, w, h);

      // Spawn signals
      if (now - stateRef.current.lastSignalTime > 2500 + Math.random() * 2000) {
        spawnSignal();
        stateRef.current.lastSignalTime = now;
      }

      // Update signals
      for (const sig of signals) {
        if (!sig.active) continue;
        sig.progress += sig.speed;
        const totalSegments = sig.path.length - 1;
        const currentPos = sig.progress * totalSegments;

        if (!sig.isBackward) {
          const activatedIndex = Math.floor(currentPos);
          for (let i = 0; i <= Math.min(activatedIndex, sig.path.length - 1); i++) {
            nodes[sig.path[i]].activation = Math.max(nodes[sig.path[i]].activation, 0.6 - i * 0.03);
          }
          for (let i = 0; i < totalSegments; i++) {
            const edgeIdx = edges.findIndex((e) => e.from === sig.path[i] && e.to === sig.path[i + 1]);
            if (edgeIdx >= 0 && currentPos > i) {
              edges[edgeIdx].forwardProgress = Math.min(1, currentPos - i);
            }
          }
          if (sig.progress >= 1) { sig.progress = 0; sig.isBackward = true; }
        } else {
          const reversedPos = sig.progress * totalSegments;
          const activatedIndex = totalSegments - Math.floor(reversedPos);
          for (let i = sig.path.length - 1; i >= Math.max(activatedIndex, 0); i--) {
            nodes[sig.path[i]].activation = Math.max(nodes[sig.path[i]].activation, 0.35);
          }
          for (let i = totalSegments - 1; i >= 0; i--) {
            const edgeIdx = edges.findIndex((e) => e.from === sig.path[i] && e.to === sig.path[i + 1]);
            if (edgeIdx >= 0 && reversedPos > (totalSegments - 1 - i)) {
              edges[edgeIdx].backwardProgress = Math.min(1, reversedPos - (totalSegments - 1 - i));
            }
          }
          if (sig.progress >= 1) sig.active = false;
        }
      }

      // Decay
      for (const node of nodes) node.activation *= 0.97;
      for (const edge of edges) {
        edge.forwardProgress *= 0.97;
        edge.backwardProgress *= 0.97;
      }
      stateRef.current.signals = signals.filter((s) => s.active || s.progress < 1.5);

      // ---- Draw layer labels at top ----
      ctx!.font = "9px monospace";
      ctx!.textAlign = "center";
      for (const ll of layerLabels) {
        ctx!.fillStyle = `rgba(${colors.label},0.25)`;
        ctx!.fillText(ll.label, ll.x, h * 0.05);
      }
      ctx!.textAlign = "start";

      // ---- Draw edges ----
      for (const edge of edges) {
        const from = nodes[edge.from];
        const to = nodes[edge.to];

        // Base edge
        ctx!.beginPath();
        ctx!.moveTo(from.x, from.y);
        ctx!.lineTo(to.x, to.y);
        ctx!.strokeStyle = `rgba(${colors.edge},0.035)`;
        ctx!.lineWidth = 0.5;
        ctx!.stroke();

        // Forward pulse
        if (edge.forwardProgress > 0.01) {
          const ex = from.x + (to.x - from.x) * Math.min(edge.forwardProgress, 1);
          const ey = from.y + (to.y - from.y) * Math.min(edge.forwardProgress, 1);
          ctx!.beginPath();
          ctx!.moveTo(from.x, from.y);
          ctx!.lineTo(ex, ey);
          ctx!.strokeStyle = `rgba(${colors.forward},${edge.forwardProgress * 0.12})`;
          ctx!.lineWidth = 1;
          ctx!.stroke();
        }

        // Backward pulse
        if (edge.backwardProgress > 0.01) {
          const ex = to.x + (from.x - to.x) * Math.min(edge.backwardProgress, 1);
          const ey = to.y + (from.y - to.y) * Math.min(edge.backwardProgress, 1);
          ctx!.beginPath();
          ctx!.moveTo(to.x, to.y);
          ctx!.lineTo(ex, ey);
          ctx!.strokeStyle = `rgba(${colors.backward},${edge.backwardProgress * 0.08})`;
          ctx!.lineWidth = 0.7;
          ctx!.setLineDash([3, 3]);
          ctx!.stroke();
          ctx!.setLineDash([]);
        }
      }

      // ---- Draw nodes ----
      for (const node of nodes) {
        const alpha = node.baseAlpha + node.activation * 0.45;

        // Glow on activation
        if (node.activation > 0.08) {
          ctx!.beginPath();
          ctx!.arc(node.x, node.y, node.radius + 4 + node.activation * 3, 0, Math.PI * 2);
          ctx!.fillStyle = `rgba(${colors.node},${node.activation * 0.06})`;
          ctx!.fill();
        }

        // Node
        ctx!.beginPath();
        ctx!.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${colors.node},${alpha})`;
        ctx!.fill();
      }

      if (!prefersReducedMotion) {
        rafRef.current = requestAnimationFrame(draw);
      }
    }

    resize();
    buildNetwork();
    draw();

    const handleResize = () => { resize(); buildNetwork(); };
    window.addEventListener("resize", handleResize);
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none"
      aria-hidden="true"
    />
  );
}
