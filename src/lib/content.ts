import fs from "fs";
import path from "path";
import matter from "gray-matter";
import readingTime from "reading-time";
import type { Post, PostFrontmatter } from "@/types/content";

const contentDirectory = path.join(process.cwd(), "content");

function getPostsFromDir(dir: string): Post[] {
  const fullPath = path.join(contentDirectory, dir);
  if (!fs.existsSync(fullPath)) return [];

  const slugs = fs.readdirSync(fullPath).filter((name) => {
    const stat = fs.statSync(path.join(fullPath, name));
    return stat.isDirectory();
  });

  const posts: Post[] = [];

  for (const slug of slugs) {
    const filePath = path.join(fullPath, slug, "index.mdx");
    if (!fs.existsSync(filePath)) continue;

    const fileContent = fs.readFileSync(filePath, "utf-8");
    const { data, content } = matter(fileContent);
    const frontmatter = data as PostFrontmatter;

    if (frontmatter.draft) continue;

    const stats = readingTime(content);

    posts.push({
      slug,
      frontmatter: {
        ...frontmatter,
        author: frontmatter.author || "Tanul Singh",
      },
      content,
      readingTime: stats.text,
    });
  }

  return posts.sort(
    (a, b) =>
      new Date(b.frontmatter.date).getTime() -
      new Date(a.frontmatter.date).getTime()
  );
}

export function getAllPosts(): Post[] {
  return getPostsFromDir("blog");
}

export function getPostBySlug(slug: string): Post | undefined {
  return getAllPosts().find((p) => p.slug === slug);
}

export function getAllPapers(): Post[] {
  return getPostsFromDir("papers");
}

export function getPaperBySlug(slug: string): Post | undefined {
  return getAllPapers().find((p) => p.slug === slug);
}

export function getFeaturedPosts(): Post[] {
  return getAllPosts().filter((p) => p.frontmatter.featured);
}

export function getAllTags(): string[] {
  const posts = [...getAllPosts(), ...getAllPapers()];
  const tags = new Set<string>();
  posts.forEach((p) => p.frontmatter.tags.forEach((t) => tags.add(t)));
  return Array.from(tags).sort();
}
