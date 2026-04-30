"use client";

import { Trophy, ScrollText, Cpu, Flame } from "lucide-react";
import { highlights } from "@/lib/constants";

const iconMap: Record<string, React.ReactNode> = {
  trophy: <Trophy size={22} />,
  scroll: <ScrollText size={22} />,
  cpu: <Cpu size={22} />,
  flame: <Flame size={22} />,
};

export function HighlightsBar() {
  return (
    <section
      className="border-y py-16"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {highlights.map((item) => (
            <div key={item.label} className="notebook-cell flex items-start gap-4 p-5">
              <div
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded"
                style={{ backgroundColor: "var(--tag-bg)", color: "var(--primary)" }}
              >
                {iconMap[item.icon]}
              </div>
              <div>
                <h3 className="font-semibold" style={{ color: "var(--text)" }}>
                  {item.label}
                </h3>
                <p className="mt-0.5 text-sm" style={{ color: "var(--text-muted)" }}>
                  {item.detail}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
