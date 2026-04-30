import type { Metadata } from "next";
import { getAllPapers } from "@/lib/content";
import { PostCard } from "@/components/blog/PostCard";

export const metadata: Metadata = {
  title: "Paper Explanations",
  description:
    "Breaking down ML research papers into intuitive explanations with diagrams and examples.",
};

export default function PapersPage() {
  const papers = getAllPapers();

  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <div className="mb-12">
        <p
          className="mb-2 text-sm font-semibold uppercase tracking-[0.15em]"
          style={{ color: "var(--primary)" }}
        >
          Papers
        </p>
        <h1 className="mb-3 text-4xl font-bold tracking-tight md:text-5xl">
          Paper Explanations
        </h1>
        <p
          className="max-w-2xl text-lg"
          style={{ color: "var(--text-secondary)" }}
        >
          ML research papers broken down into intuitive explanations — making
          cutting-edge research accessible to everyone.
        </p>
      </div>

      <div className="grid gap-6">
        {papers.map((paper) => (
          <PostCard key={paper.slug} post={paper} basePath="/papers" />
        ))}
      </div>

      {papers.length === 0 && (
        <div
          className="rounded-xl border border-dashed p-12 text-center"
          style={{ borderColor: "var(--border)" }}
        >
          <p className="text-lg" style={{ color: "var(--text-muted)" }}>
            Paper explanations coming soon.
          </p>
          <p className="mt-2 text-sm" style={{ color: "var(--text-muted)" }}>
            In the meantime, check out the{" "}
            <a href="/blog" style={{ color: "var(--primary)" }} className="underline">
              blog
            </a>{" "}
            for existing deep dives.
          </p>
        </div>
      )}
    </div>
  );
}
