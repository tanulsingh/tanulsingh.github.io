import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getAllPapers, getPaperBySlug } from "@/lib/content";
import { mdxOptions } from "@/lib/mdx";
import { mdxComponents } from "@/components/mdx/MdxComponents";
import { PostHeader } from "@/components/blog/PostHeader";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export const dynamicParams = false;

export async function generateStaticParams() {
  return getAllPapers().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const paper = getPaperBySlug(slug);
  if (!paper) return {};

  return {
    title: paper.frontmatter.title,
    description: paper.frontmatter.description || paper.frontmatter.summary,
  };
}

export default async function PaperPage({ params }: PageProps) {
  const { slug } = await params;
  const paper = getPaperBySlug(slug);

  if (!paper) notFound();

  return (
    <article className="mx-auto max-w-3xl px-6 pb-24 pt-32">
      <PostHeader
        title={paper.frontmatter.title}
        date={paper.frontmatter.date}
        readingTime={paper.readingTime}
        tags={paper.frontmatter.tags}
        coverImage={paper.frontmatter.coverImage}
        coverAlt={paper.frontmatter.coverAlt}
      />

      <div className="prose prose-lg prose-paper max-w-none dark:prose-invert">
        <MDXRemote
          source={paper.content}
          components={mdxComponents}
          options={mdxOptions}
        />
      </div>
    </article>
  );
}
