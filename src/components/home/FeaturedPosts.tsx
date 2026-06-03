import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { formatDate } from "@/lib/utils";
import type { Post } from "@/types/content";

interface FeaturedPostsProps {
  posts: Post[];
}

export function FeaturedPosts({ posts }: FeaturedPostsProps) {
  if (posts.length === 0) return null;

  return (
    <section className="lab-page py-24">
      <div className="mx-auto max-w-3xl px-6">
        <div className="mb-8 flex items-end justify-between">
          <div>
            <p
              className="lab-meta mb-2 text-xs uppercase tracking-widest"
              style={{ color: "var(--primary)" }}
            >
              latest
            </p>
            <h2
              className="text-3xl font-bold tracking-tight md:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Recent writing
            </h2>
          </div>
          <Link
            href="/blog"
            className="group hidden items-center gap-1 font-mono text-xs transition-colors hover:opacity-80 md:flex"
            style={{ color: "var(--text-secondary)" }}
          >
            view all
            <ArrowRight size={12} className="transition-transform group-hover:translate-x-1" />
          </Link>
        </div>

        <div>
          {posts.map((post) => (
            <Link
              key={post.slug}
              href={`${post.basePath}/${post.slug}`}
              className="lab-row group block"
            >
              <span className="lab-date">
                {formatDate(post.frontmatter.date)}
              </span>
              <div className="lab-title group-hover:opacity-80">
                {post.frontmatter.title}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
