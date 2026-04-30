import type { Metadata } from "next";
import { getAllPosts } from "@/lib/content";
import { PostCard } from "@/components/blog/PostCard";

export const metadata: Metadata = {
  title: "Blog",
  description:
    "Deep dives into machine learning, NLP, transformer architectures, and AI research.",
};

export default function BlogPage() {
  const posts = getAllPosts();

  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <div className="mb-12">
        <p className="mb-1 font-mono text-sm" style={{ color: "var(--primary)" }}>
          blog/
        </p>
        <h1
          className="mb-3 text-4xl font-bold tracking-tight md:text-5xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Articles &amp; Deep Dives
        </h1>
        <p
          className="max-w-2xl text-lg"
          style={{ color: "var(--text-secondary)" }}
        >
          Explorations of ML papers, transformer architectures, and lessons
          learned building AI systems in production.
        </p>
      </div>

      <div className="grid gap-6">
        {posts.map((post) => (
          <PostCard key={post.slug} post={post} />
        ))}
      </div>

      {posts.length === 0 && (
        <p className="text-center" style={{ color: "var(--text-muted)" }}>
          No posts yet. Check back soon!
        </p>
      )}
    </div>
  );
}
