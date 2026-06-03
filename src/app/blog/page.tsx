import type { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import { getAllPosts } from "@/lib/content";
import { HIDDEN_TAGS } from "@/lib/constants";
import { SeriesCascade } from "@/components/blog/SeriesCascade";
import { formatDate } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Field Notes",
  description:
    "Written notes from my attempts to satisfy my curious mind — long series, lab notes, and foundational deep dives.",
};

const SERIES = [
  {
    id: "gradient-descent-transformers",
    title: "The Gradient Descent through Transformers",
    blurb:
      "A component-by-component rebuild of the modern transformer stack — tokenization, attention, positional encoding, FlashAttention, the rest of it. Aimed at anyone who wants to understand the architecture from the ground up.",
    caption: "Each Transformer component, traced in detail from 2017 to now.",
    tag: "The Gradient Descent through Transformers",
    image: "/images/blog/gradient-descent-transformers/cover.png",
  },
  {
    id: "loss-landscape-llm-training",
    title: "The Loss Landscape of LLM Training",
    blurb:
      "How LLMs are actually trained end-to-end — pre-training, distributed systems, mixed precision, PEFT, SFT, RLHF, DPO, and the reasoning + agents work that came after.",
    caption: "Pre-training, fine-tuning, post-training — how LLMs are actually trained.",
    tag: "The Loss Landscape of LLM Training",
    image: "/images/blog/loss-landscape-llm-training/cover.png",
  },
];

// Tags whose posts should be hidden from the index entirely (e.g., draft series
// not yet ready to publish). Posts with any of these tags are filtered before
// series and standalone lists are computed.
// Imported from @/lib/constants so blog, papers, and homepage all stay in sync.

export default function BlogPage() {
  const allPosts = getAllPosts();
  const posts = allPosts
    .filter((p) => !p.frontmatter.tags.some((t) => HIDDEN_TAGS.includes(t)))
    .filter((p) => !p.frontmatter.draft);

  const seriesWithPosts = SERIES.map((series) => ({
    ...series,
    posts: posts
      .filter((p) => p.frontmatter.tags.includes(series.tag))
      .sort(
        (a, b) =>
          new Date(a.frontmatter.date).getTime() -
          new Date(b.frontmatter.date).getTime()
      ),
  }));

  const standalone = posts
    .filter((p) => !SERIES.some((s) => p.frontmatter.tags.includes(s.tag)))
    .sort(
      (a, b) =>
        new Date(b.frontmatter.date).getTime() -
        new Date(a.frontmatter.date).getTime()
    );

  return (
    <div className="lab-page mx-auto max-w-3xl px-6 pb-24 pt-32">
      <header className="mb-12">
        <p className="lab-meta mb-2 text-xs uppercase tracking-widest" style={{ color: "var(--primary)" }}>
          blog/
        </p>
        <h1
          className="mb-3 text-3xl font-bold tracking-tight md:text-4xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Field Notes
        </h1>
        <p
          className="max-w-2xl text-sm leading-relaxed"
          style={{ color: "var(--text-secondary)", fontFamily: "var(--font-serif)" }}
        >
          Written notes from my attempts to satisfy my curious mind.
        </p>
      </header>

      <section className="mb-14">
        <h2 className="lab-section-label">Series</h2>
        <div>
          {seriesWithPosts.map((s, i) => (
            <SeriesCascade
              key={s.id}
              marker={`series ${String(i + 1).padStart(2, "0")}`}
              title={s.title}
              blurb={s.blurb}
              caption={s.caption}
              meta={
                s.posts.length > 0
                  ? `${s.posts.length} ${s.posts.length === 1 ? "part shipped" : "parts shipped"}`
                  : "coming soon"
              }
              posts={s.posts}
              image={s.image}
            />
          ))}
        </div>
      </section>

      {standalone.length > 0 && (
        <section>
          <h2 className="lab-section-label">Standalone</h2>
          <div>
            {standalone.map((post) => {
              const cover = post.frontmatter.coverImage;
              const caption = post.frontmatter.coverCaption;
              return (
                <Link
                  key={post.slug}
                  href={`/blog/${post.slug}`}
                  className={`lab-row group block${cover ? " has-thumb" : ""}`}
                >
                  <span className="lab-date">
                    {formatDate(post.frontmatter.date)}
                  </span>
                  {cover && (
                    <span className="lab-thumb">
                      <Image
                        src={cover}
                        alt={post.frontmatter.coverAlt || post.frontmatter.title}
                        width={160}
                        height={160}
                      />
                    </span>
                  )}
                  <div>
                    <div className="lab-title group-hover:opacity-80">
                      {post.frontmatter.title}
                    </div>
                    {cover && caption && (
                      <p className="lab-caption">{caption}</p>
                    )}
                    {!cover && (
                      <p className="lab-blurb">
                        {post.frontmatter.summary || post.frontmatter.description}
                      </p>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      )}

      {posts.length === 0 && (
        <p
          className="text-center text-sm"
          style={{ color: "var(--text-muted)", fontFamily: "var(--font-serif)" }}
        >
          Nothing here yet.
        </p>
      )}
    </div>
  );
}
