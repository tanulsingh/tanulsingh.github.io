import type { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import { getAllPapers } from "@/lib/content";
import { HIDDEN_TAGS } from "@/lib/constants";
import { formatDate } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Papers, Annotated",
  description:
    "Reading notes — ML papers broken down with diagrams, derivations, and the parts that took me longest to understand.",
};

export default function PapersPage() {
  const papers = getAllPapers()
    .filter(
      (p) => !p.frontmatter.tags.some((t) => HIDDEN_TAGS.includes(t))
    )
    .filter((p) => !p.frontmatter.draft)
    .sort(
      (a, b) =>
        new Date(b.frontmatter.date).getTime() -
        new Date(a.frontmatter.date).getTime()
    );

  return (
    <div className="lab-page mx-auto max-w-3xl px-6 pb-24 pt-32">
      <header className="mb-12">
        <p
          className="lab-meta mb-2 text-xs uppercase tracking-widest"
          style={{ color: "var(--primary)" }}
        >
          papers/
        </p>
        <h1
          className="mb-3 text-3xl font-bold tracking-tight md:text-4xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Papers, Annotated
        </h1>
        <p
          className="max-w-2xl text-sm leading-relaxed"
          style={{
            color: "var(--text-secondary)",
            fontFamily: "var(--font-serif)",
          }}
        >
          Reading notes for papers I&apos;ve worked through — the diagrams I
          wished existed, the derivations I had to redo, and the parts that took
          me longest to understand.
        </p>
      </header>

      {papers.length > 0 ? (
        <section>
          <h2 className="lab-section-label">Annotated</h2>
          <div>
            {papers.map((paper) => {
              const cover = paper.frontmatter.coverImage;
              const caption = paper.frontmatter.coverCaption;
              return (
                <Link
                  key={paper.slug}
                  href={`/papers/${paper.slug}`}
                  className={`lab-row group block${cover ? " has-thumb" : ""}`}
                >
                  <span className="lab-date">
                    {formatDate(paper.frontmatter.date)}
                  </span>
                  {cover && (
                    <span className="lab-thumb">
                      <Image
                        src={cover}
                        alt={
                          paper.frontmatter.coverAlt || paper.frontmatter.title
                        }
                        width={160}
                        height={160}
                      />
                    </span>
                  )}
                  <div>
                    <div className="lab-title group-hover:opacity-80">
                      {paper.frontmatter.title}
                    </div>
                    {cover && caption && (
                      <p className="lab-caption">{caption}</p>
                    )}
                    {!cover && (
                      <p className="lab-blurb">
                        {paper.frontmatter.summary ||
                          paper.frontmatter.description}
                      </p>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      ) : (
        <p
          className="text-sm"
          style={{
            color: "var(--text-muted)",
            fontFamily: "var(--font-serif)",
            fontStyle: "italic",
          }}
        >
          Nothing here yet — annotated reads coming soon.
        </p>
      )}
    </div>
  );
}
