"use client";

import { useEffect, useRef } from "react";

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
    width: number;
    height: number;
  }>({ nodes: [], edges: [], signals: [], lastSignalTime: 0, width: 0, height: 0 });

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

      const layerCount = isMobile ? 4 : 6;
      const nodesPerLayer = isMobile ? [3, 5, 6, 4] : [4, 6, 8, 8, 6, 3];
      const nodes: NetNode[] = [];
      const edges: Edge[] = [];

      const padX = w * 0.08;
      const padY = h * 0.12;
      const usableW = w - padX * 2;
      const usableH = h - padY * 2;

      for (let l = 0; l < layerCount; l++) {
        const count = nodesPerLayer[l];
        const x = padX + (l / (layerCount - 1)) * usableW;

        for (let n = 0; n < count; n++) {
          const ySpread = usableH * 0.7;
          const yOffset = (usableH - ySpread) / 2;
          const y = padY + yOffset + (n / (count - 1)) * ySpread;

          const jitterX = (Math.random() - 0.5) * usableW * 0.03;
          const jitterY = (Math.random() - 0.5) * ySpread * 0.08;

          nodes.push({
            x: x + jitterX,
            y: y + jitterY,
            layer: l,
            radius: isMobile ? 2 : 2.5,
            baseAlpha: 0.08 + Math.random() * 0.06,
            activation: 0,
          });
        }
      }

      for (let i = 0; i < nodes.length; i++) {
        for (let j = 0; j < nodes.length; j++) {
          if (nodes[j].layer !== nodes[i].layer + 1) continue;
          if (Math.random() > 0.6) continue;
          edges.push({ from: i, to: j, forwardProgress: 0, backwardProgress: 0 });
        }
      }

      stateRef.current.nodes = nodes;
      stateRef.current.edges = edges;
      stateRef.current.signals = [];
      stateRef.current.lastSignalTime = performance.now();
    }

    function findPathForward(nodes: NetNode[], edges: Edge[]): number[] {
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
      const path = findPathForward(nodes, edges);
      if (path.length < 2) return;

      stateRef.current.signals.push({
        path,
        progress: 0,
        speed: 0.008 + Math.random() * 0.006,
        isBackward: false,
        active: true,
      });
    }

    function draw() {
      const { nodes, edges, signals, width: w, height: h } = stateRef.current;
      const colors = getColors();
      const now = performance.now();

      ctx!.clearRect(0, 0, w, h);

      if (now - stateRef.current.lastSignalTime > 3000 + Math.random() * 2000) {
        spawnSignal();
        stateRef.current.lastSignalTime = now;
      }

      for (const sig of signals) {
        if (!sig.active) continue;
        sig.progress += sig.speed;

        const totalSegments = sig.path.length - 1;
        const currentPos = sig.progress * totalSegments;

        if (!sig.isBackward) {
          const activatedIndex = Math.floor(currentPos);
          for (let i = 0; i <= Math.min(activatedIndex, sig.path.length - 1); i++) {
            const nodeIdx = sig.path[i];
            nodes[nodeIdx].activation = Math.max(nodes[nodeIdx].activation, 0.7 - i * 0.05);
          }

          for (let i = 0; i < totalSegments; i++) {
            const edgeIdx = edges.findIndex(
              (e) => e.from === sig.path[i] && e.to === sig.path[i + 1]
            );
            if (edgeIdx >= 0 && currentPos > i) {
              edges[edgeIdx].forwardProgress = Math.min(1, (currentPos - i));
            }
          }

          if (sig.progress >= 1) {
            sig.progress = 0;
            sig.isBackward = true;
          }
        } else {
          const reversedPos = sig.progress * totalSegments;
          const activatedIndex = totalSegments - Math.floor(reversedPos);

          for (let i = sig.path.length - 1; i >= Math.max(activatedIndex, 0); i--) {
            const nodeIdx = sig.path[i];
            nodes[nodeIdx].activation = Math.max(nodes[nodeIdx].activation, 0.4);
          }

          for (let i = totalSegments - 1; i >= 0; i--) {
            const edgeIdx = edges.findIndex(
              (e) => e.from === sig.path[i] && e.to === sig.path[i + 1]
            );
            if (edgeIdx >= 0 && reversedPos > (totalSegments - 1 - i)) {
              edges[edgeIdx].backwardProgress = Math.min(1, reversedPos - (totalSegments - 1 - i));
            }
          }

          if (sig.progress >= 1) {
            sig.active = false;
          }
        }
      }

      for (const node of nodes) {
        node.activation *= 0.97;
      }
      for (const edge of edges) {
        edge.forwardProgress *= 0.97;
        edge.backwardProgress *= 0.97;
      }

      stateRef.current.signals = signals.filter((s) => s.active || s.progress < 1.5);

      // Draw edges
      for (const edge of edges) {
        const from = nodes[edge.from];
        const to = nodes[edge.to];

        ctx!.beginPath();
        ctx!.moveTo(from.x, from.y);
        ctx!.lineTo(to.x, to.y);
        ctx!.strokeStyle = `rgba(${colors.edge},0.025)`;
        ctx!.lineWidth = 0.4;
        ctx!.stroke();

        // Forward signal
        if (edge.forwardProgress > 0.01) {
          const ex = from.x + (to.x - from.x) * Math.min(edge.forwardProgress, 1);
          const ey = from.y + (to.y - from.y) * Math.min(edge.forwardProgress, 1);
          ctx!.beginPath();
          ctx!.moveTo(from.x, from.y);
          ctx!.lineTo(ex, ey);
          ctx!.strokeStyle = `rgba(${colors.forward},${edge.forwardProgress * 0.08})`;
          ctx!.lineWidth = 0.8;
          ctx!.stroke();
        }

        // Backward signal
        if (edge.backwardProgress > 0.01) {
          const ex = to.x + (from.x - to.x) * Math.min(edge.backwardProgress, 1);
          const ey = to.y + (from.y - to.y) * Math.min(edge.backwardProgress, 1);
          ctx!.beginPath();
          ctx!.moveTo(to.x, to.y);
          ctx!.lineTo(ex, ey);
          ctx!.strokeStyle = `rgba(${colors.backward},${edge.backwardProgress * 0.06})`;
          ctx!.lineWidth = 0.5;
          ctx!.setLineDash([3, 3]);
          ctx!.stroke();
          ctx!.setLineDash([]);
        }
      }

      // Draw nodes
      for (const node of nodes) {
        const alpha = node.baseAlpha + node.activation * 0.5;

        if (node.activation > 0.1) {
          ctx!.beginPath();
          ctx!.arc(node.x, node.y, node.radius + 3 + node.activation * 2, 0, Math.PI * 2);
          ctx!.fillStyle = `rgba(${colors.node},${node.activation * 0.04})`;
          ctx!.fill();
        }

        ctx!.beginPath();
        ctx!.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${colors.node},${alpha * 0.6})`;
        ctx!.fill();
      }

      if (!prefersReducedMotion) {
        rafRef.current = requestAnimationFrame(draw);
      }
    }

    resize();
    buildNetwork();
    draw();

    const handleResize = () => {
      resize();
      buildNetwork();
    };

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
