import Link from "next/link";
import { ArrowRight, Calendar, Clock } from "lucide-react";
import { formatDate } from "@/lib/utils";
import type { Post } from "@/types/content";

interface FeaturedPostsProps {
  posts: Post[];
}

export function FeaturedPosts({ posts }: FeaturedPostsProps) {
  if (posts.length === 0) return null;

  return (
    <section className="py-24">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mb-12 flex items-end justify-between">
          <div>
            <p className="mb-1 font-mono text-sm" style={{ color: "var(--primary)" }}>
              latest
            </p>
            <h2
              className="text-3xl font-bold tracking-tight md:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Paper Explanations &amp; Articles
            </h2>
          </div>
          <Link
            href="/blog"
            className="group hidden items-center gap-1 text-sm font-medium transition-colors hover:opacity-80 md:flex"
            style={{ color: "var(--text-secondary)" }}
          >
            View all
            <ArrowRight size={14} className="transition-transform group-hover:translate-x-1" />
          </Link>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {posts.map((post) => (
            <Link key={post.slug} href={`${post.basePath}/${post.slug}`} className="group block">
              <article className="notebook-cell h-full overflow-hidden">
                {post.frontmatter.coverImage && (
                  <div className="aspect-video overflow-hidden">
                    <img
                      src={post.frontmatter.coverImage}
                      alt={post.frontmatter.coverAlt || post.frontmatter.title}
                      className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                    />
                  </div>
                )}
                <div className="p-6">
                  <div className="mb-3 flex flex-wrap gap-2">
                    {post.frontmatter.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="rounded px-2 py-0.5 font-mono text-xs"
                        style={{ backgroundColor: "var(--tag-bg)", color: "var(--tag-text)" }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <h3
                    className="mb-2 text-xl font-bold group-hover:opacity-80"
                    style={{ fontFamily: "var(--font-serif)" }}
                  >
                    {post.frontmatter.title}
                  </h3>
                  <p className="mb-4 line-clamp-2 text-sm" style={{ color: "var(--text-secondary)" }}>
                    {post.frontmatter.summary || post.frontmatter.description}
                  </p>
                  <div className="flex items-center gap-4 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
                    <span className="flex items-center gap-1">
                      <Calendar size={12} />
                      {formatDate(post.frontmatter.date)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock size={12} />
                      {post.readingTime}
                    </span>
                  </div>
                </div>
              </article>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
