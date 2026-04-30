import Link from "next/link";
import { Calendar, Clock } from "lucide-react";
import { formatDate } from "@/lib/utils";
import type { Post } from "@/types/content";

interface PostCardProps {
  post: Post;
  basePath?: string;
}

export function PostCard({ post, basePath = "/blog" }: PostCardProps) {
  return (
    <Link href={`${basePath}/${post.slug}`} className="group block">
      <article className="notebook-cell p-6">
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
          className="mb-2 text-lg font-bold group-hover:opacity-80"
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
      </article>
    </Link>
  );
}
