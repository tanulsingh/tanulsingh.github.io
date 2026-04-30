import { Calendar, Clock, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { formatDate } from "@/lib/utils";

interface PostHeaderProps {
  title: string;
  date: string;
  readingTime: string;
  tags: string[];
  coverImage?: string;
  coverAlt?: string;
}

export function PostHeader({ title, date, readingTime, tags, coverImage, coverAlt }: PostHeaderProps) {
  return (
    <header className="mb-10">
      <Link
        href="/blog"
        className="mb-8 inline-flex items-center gap-1.5 font-mono text-sm transition-colors hover:opacity-80"
        style={{ color: "var(--text-muted)" }}
      >
        <ArrowLeft size={14} />
        cd ../blog
      </Link>

      <div className="mb-4 flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span
            key={tag}
            className="rounded px-2.5 py-1 font-mono text-xs"
            style={{ backgroundColor: "var(--tag-bg)", color: "var(--tag-text)" }}
          >
            {tag}
          </span>
        ))}
      </div>

      <h1
        className="mb-4 text-3xl font-bold leading-tight tracking-tight md:text-5xl"
        style={{ fontFamily: "var(--font-serif)" }}
      >
        {title}
      </h1>

      <div className="flex items-center gap-4 font-mono text-sm" style={{ color: "var(--text-muted)" }}>
        <span className="flex items-center gap-1.5">
          <Calendar size={14} />
          {formatDate(date)}
        </span>
        <span className="flex items-center gap-1.5">
          <Clock size={14} />
          {readingTime}
        </span>
      </div>

      {coverImage && (
        <div className="mt-8 overflow-hidden rounded" style={{ border: "1px solid var(--border)" }}>
          <img src={coverImage} alt={coverAlt || title} className="w-full object-cover" />
        </div>
      )}
    </header>
  );
}
