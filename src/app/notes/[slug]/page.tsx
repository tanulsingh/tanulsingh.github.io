import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getAllNotes, getNoteBySlug } from "@/lib/content";
import { mdxOptions } from "@/lib/mdx";
import { mdxComponents } from "@/components/mdx/MdxComponents";
import { PostHeader } from "@/components/blog/PostHeader";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export const dynamicParams = false;

export async function generateStaticParams() {
  return getAllNotes().map((note) => ({ slug: note.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const note = getNoteBySlug(slug);
  if (!note) return {};

  return {
    title: note.frontmatter.title,
    description: note.frontmatter.description || note.frontmatter.summary,
    openGraph: {
      title: note.frontmatter.title,
      description: note.frontmatter.description || note.frontmatter.summary,
      type: "article",
      publishedTime: note.frontmatter.date,
      tags: note.frontmatter.tags,
    },
  };
}

export default async function NotePage({ params }: PageProps) {
  const { slug } = await params;
  const note = getNoteBySlug(slug);

  if (!note) notFound();

  return (
    <article className="mx-auto max-w-3xl px-6 pb-24 pt-32">
      <Link
        href="/notes"
        className="mb-6 inline-block font-mono text-xs hover:opacity-80"
        style={{ color: "var(--primary)" }}
      >
        ← notes/
      </Link>

      <PostHeader
        title={note.frontmatter.title}
        date={note.frontmatter.date}
        readingTime={note.readingTime}
        tags={note.frontmatter.tags}
        coverImage={note.frontmatter.coverImage}
        coverAlt={note.frontmatter.coverAlt}
      />

      <div className="prose prose-lg prose-paper max-w-none dark:prose-invert">
        <MDXRemote
          source={note.content}
          components={mdxComponents}
          options={mdxOptions}
        />
      </div>
    </article>
  );
}
