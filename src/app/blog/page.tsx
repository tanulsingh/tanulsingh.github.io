import type { Metadata } from "next";
import { getAllPosts } from "@/lib/content";
import { PostCard } from "@/components/blog/PostCard";

export const metadata: Metadata = {
  title: "Blog",
  description:
    "Deep dives into machine learning, NLP, transformer architectures, and AI research.",
};

const SERIES = [
  {
    id: "gradient-descent-transformers",
    title: "The Gradient Descent through Transformers",
    description:
      "A component-by-component deep dive into the modern transformer stack — how every piece evolved from 2017 to 2026, and why each one matters.",
    tag: "The Gradient Descent through Transformers",
  },
];

export default function BlogPage() {
  const posts = getAllPosts();

  const seriesPosts = SERIES.map((series) => ({
    ...series,
    posts: posts
      .filter((p) => p.frontmatter.tags.includes(series.tag))
      .sort((a, b) => new Date(a.frontmatter.date).getTime() - new Date(b.frontmatter.date).getTime()),
  }));

  const standalonePosts = posts.filter(
    (p) => !SERIES.some((s) => p.frontmatter.tags.includes(s.tag))
  );

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

      {/* Series */}
      {seriesPosts.map((series) => (
        <div key={series.id} className="mb-16">
          <div
            className="mb-6 rounded border-l-3 p-5"
            style={{
              borderColor: "var(--primary)",
              backgroundColor: "var(--bg-surface)",
            }}
          >
            <p className="mb-1 font-mono text-xs" style={{ color: "var(--primary)" }}>
              series
            </p>
            <h2
              className="mb-2 text-2xl font-bold tracking-tight"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              {series.title}
            </h2>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              {series.description}
            </p>
          </div>

          <div className="grid gap-4">
            {series.posts.map((post, i) => (
              <div key={post.slug} className="flex gap-4">
                <span
                  className="mt-5 shrink-0 font-mono text-xs tabular-nums"
                  style={{ color: "var(--text-muted)", minWidth: "2rem" }}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className="flex-1">
                  <PostCard post={post} />
                </div>
              </div>
            ))}
          </div>

          {series.posts.length === 0 && (
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              Posts coming soon.
            </p>
          )}
        </div>
      ))}

      {/* Standalone posts */}
      {standalonePosts.length > 0 && (
        <div>
          {seriesPosts.some((s) => s.posts.length > 0) && (
            <h2
              className="mb-6 text-2xl font-bold tracking-tight"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Other Articles
            </h2>
          )}
          <div className="grid gap-6">
            {standalonePosts.map((post) => (
              <PostCard key={post.slug} post={post} />
            ))}
          </div>
        </div>
      )}

      {posts.length === 0 && (
        <p className="text-center" style={{ color: "var(--text-muted)" }}>
          No posts yet. Check back soon!
        </p>
      )}
    </div>
  );
}
