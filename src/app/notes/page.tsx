import type { Metadata } from "next";
import Link from "next/link";
import { getAllNotes } from "@/lib/content";
import { formatDate } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Notes",
  description:
    "Short notes capturing the intuitions that clicked while building things.",
};

export default function NotesPage() {
  const notes = getAllNotes().filter((n) => !n.frontmatter.draft);

  return (
    <div className="lab-page mx-auto max-w-3xl px-6 pb-24 pt-32">
      <header className="mb-12">
        <p
          className="lab-meta mb-2 text-xs uppercase tracking-widest"
          style={{ color: "var(--primary)" }}
        >
          notes/
        </p>
        <h1
          className="mb-3 text-3xl font-bold tracking-tight md:text-4xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Notes
        </h1>
        <p
          className="max-w-2xl text-sm leading-relaxed"
          style={{
            color: "var(--text-secondary)",
            fontFamily: "var(--font-serif)",
          }}
        >
          Short notes capturing the intuitions that clicked while building
          things. Usually one question I had and what I figured out.
        </p>
      </header>

      {notes.length > 0 ? (
        <section>
          <div>
            {notes.map((note) => (
              <Link
                key={note.slug}
                href={`/notes/${note.slug}`}
                className="lab-row group block"
              >
                <span className="lab-date">
                  {formatDate(note.frontmatter.date)}
                </span>
                <div>
                  <div className="lab-title group-hover:opacity-80">
                    {note.frontmatter.title}
                  </div>
                  <p className="lab-blurb">
                    {note.frontmatter.summary ||
                      note.frontmatter.description}
                  </p>
                </div>
              </Link>
            ))}
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
          Nothing here yet.
        </p>
      )}
    </div>
  );
}
