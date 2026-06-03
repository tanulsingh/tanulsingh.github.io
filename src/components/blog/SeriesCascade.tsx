"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import type { Post } from "@/types/content";

interface SeriesCascadeProps {
  marker: string;
  title: string;
  blurb: string;
  caption?: string;
  meta: string;
  posts: Post[];
  image?: string;
  imageAlt?: string;
  defaultOpen?: boolean;
}

export function SeriesCascade({
  marker,
  title,
  blurb,
  caption,
  meta,
  posts,
  image,
  imageAlt,
  defaultOpen = false,
}: SeriesCascadeProps) {
  const [open, setOpen] = useState(defaultOpen);
  const hasThumb = !!image;

  return (
    <div className={`series-row${hasThumb ? " has-thumb" : ""}`} data-open={open}>
      <button
        type="button"
        className="series-trigger"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <span className="series-marker">{marker}</span>
        {hasThumb && (
          <span className="series-thumb">
            <Image
              src={image!}
              alt={imageAlt || title}
              width={160}
              height={160}
            />
          </span>
        )}
        <span>
          <span className="series-title block">{title}</span>
          {!hasThumb && <span className="series-blurb block">{blurb}</span>}
          {hasThumb && caption && (
            <span className="series-caption block">{caption}</span>
          )}
          <span className="series-meta block">{meta}</span>
        </span>
        <span className="series-chevron">{open ? "[—]" : "[+]"}</span>
      </button>

      {open && posts.length > 0 && (
        <div className="series-children">
          {posts.map((post, i) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="series-child group"
            >
              <span className="series-child-num">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="series-child-title">
                {post.frontmatter.title}
              </span>
            </Link>
          ))}
        </div>
      )}

      {open && posts.length === 0 && (
        <div className="series-children">
          <p
            className="series-child-title"
            style={{ color: "var(--text-muted)", fontStyle: "italic", paddingLeft: "1rem" }}
          >
            posts coming soon.
          </p>
        </div>
      )}
    </div>
  );
}
