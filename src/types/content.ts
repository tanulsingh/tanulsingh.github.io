export interface PostFrontmatter {
  title: string;
  date: string;
  tags: string[];
  description: string;
  summary: string;
  coverImage?: string;
  coverAlt?: string;
  featured?: boolean;
  draft?: boolean;
  author?: string;
}

export interface Post {
  slug: string;
  frontmatter: PostFrontmatter;
  content: string;
  readingTime: string;
  basePath: string;
}

export interface ProjectData {
  title: string;
  description: string;
  tags: string[];
  link?: string;
  github?: string;
  image?: string;
}
