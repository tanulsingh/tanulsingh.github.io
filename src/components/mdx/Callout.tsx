import { Info, AlertTriangle, Lightbulb } from "lucide-react";

interface CalloutProps {
  type?: "info" | "warning" | "tip";
  children: React.ReactNode;
}

const styles = {
  info: {
    border: "rgba(96, 165, 250, 0.3)",
    bg: "rgba(96, 165, 250, 0.05)",
    icon: <Info size={18} color="#60a5fa" />,
  },
  warning: {
    border: "rgba(251, 191, 36, 0.3)",
    bg: "rgba(251, 191, 36, 0.05)",
    icon: <AlertTriangle size={18} color="#fbbf24" />,
  },
  tip: {
    border: "rgba(52, 211, 153, 0.3)",
    bg: "rgba(52, 211, 153, 0.05)",
    icon: <Lightbulb size={18} color="#34d399" />,
  },
};

export function Callout({ type = "info", children }: CalloutProps) {
  const s = styles[type];

  return (
    <div
      className="my-6 flex gap-3 rounded-lg border p-4"
      style={{ borderColor: s.border, backgroundColor: s.bg }}
    >
      <div className="mt-0.5 shrink-0">{s.icon}</div>
      <div className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
        {children}
      </div>
    </div>
  );
}
