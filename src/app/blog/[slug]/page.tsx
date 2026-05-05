import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getAllPosts, getPostBySlug } from "@/lib/content";
import { mdxOptions } from "@/lib/mdx";
import { mdxComponents } from "@/components/mdx/MdxComponents";
import { PostHeader } from "@/components/blog/PostHeader";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export const dynamicParams = false;

export async function generateStaticParams() {
  return getAllPosts().map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) return {};

  return {
    title: post.frontmatter.title,
    description: post.frontmatter.description || post.frontmatter.summary,
    openGraph: {
      title: post.frontmatter.title,
      description: post.frontmatter.description || post.frontmatter.summary,
      type: "article",
      publishedTime: post.frontmatter.date,
      tags: post.frontmatter.tags,
    },
  };
}

export default async function BlogPostPage({ params }: PageProps) {
  const { slug } = await params;
  const post = getPostBySlug(slug);

  if (!post) notFound();

  const SERIES_TAG = "The Gradient Descent through Transformers";
  const isSeries = post.frontmatter.tags.includes(SERIES_TAG);
  const allPosts = getAllPosts();
  const seriesPosts = isSeries
    ? allPosts
        .filter((p) => p.frontmatter.tags.includes(SERIES_TAG))
        .sort((a, b) => new Date(a.frontmatter.date).getTime() - new Date(b.frontmatter.date).getTime())
    : [];
  const partNumber = seriesPosts.findIndex((p) => p.slug === slug) + 1;

  return (
    <article className="mx-auto max-w-3xl px-6 pb-24 pt-32">
      {isSeries && (
        <Link
          href="/blog"
          className="mb-6 block rounded border-l-3 p-4 transition-colors hover:opacity-80"
          style={{
            borderColor: "var(--primary)",
            backgroundColor: "var(--bg-surface)",
          }}
        >
          <p className="font-mono text-xs" style={{ color: "var(--primary)" }}>
            Part {partNumber} of {seriesPosts.length}
          </p>
          <p
            className="text-sm font-medium"
            style={{ fontFamily: "var(--font-serif)", color: "var(--text-secondary)" }}
          >
            The Gradient Descent through Transformers
          </p>
        </Link>
      )}

      <PostHeader
        title={post.frontmatter.title}
        date={post.frontmatter.date}
        readingTime={post.readingTime}
        tags={post.frontmatter.tags}
        coverImage={post.frontmatter.coverImage}
        coverAlt={post.frontmatter.coverAlt}
      />

      <div className="prose prose-lg prose-paper max-w-none dark:prose-invert">
        <MDXRemote
          source={post.content}
          components={mdxComponents}
          options={mdxOptions}
        />
      </div>
    </article>
  );
}
