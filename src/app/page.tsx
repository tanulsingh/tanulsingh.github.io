import { HeroSection } from "@/components/home/HeroSection";
import { HighlightsBar } from "@/components/home/HighlightsBar";
import { LearningCurve } from "@/components/home/LearningCurve";
import { FeaturedPosts } from "@/components/home/FeaturedPosts";
import { getAllPosts, getAllPapers } from "@/lib/content";

export default function Home() {
  const posts = [...getAllPosts(), ...getAllPapers()]
    .sort((a, b) => new Date(b.frontmatter.date).getTime() - new Date(a.frontmatter.date).getTime())
    .slice(0, 4);

  return (
    <>
      <HeroSection />
      <HighlightsBar />
      <LearningCurve />

      {/* Philosophy */}
      <section
        className="border-y py-20"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
      >
        <div className="mx-auto max-w-3xl px-6 text-center">
          <blockquote
            className="text-2xl font-medium italic leading-relaxed md:text-3xl"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            <span className="gradient-text">
              &ldquo;You don&rsquo;t need a low initial loss. You need a good learning rate and the patience to keep training.&rdquo;
            </span>
          </blockquote>
          <p className="mt-4 font-mono text-xs" style={{ color: "var(--text-muted)" }}>
            — the philosophy that took me from Mechanical Engineering to Apple
          </p>
        </div>
      </section>

      <FeaturedPosts posts={posts} />

      {/* CTA */}
      <section className="py-20">
        <div className="mx-auto max-w-3xl px-6 text-center">
          <h2
            className="mb-4 text-2xl font-bold md:text-3xl"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Want to know more?
          </h2>
          <p style={{ color: "var(--text-secondary)" }}>
            Watch my conversations with people who shaped the Indian ML community:
          </p>
          <div className="mt-6 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <a
              href="https://www.youtube.com/watch?v=t-kiCW0jdLg&t=490s"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border px-6 py-2.5 text-sm font-medium transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
            >
              Discussion with Dhruv (CodeBasics)
            </a>
            <a
              href="https://www.youtube.com/watch?v=ujGLgn3fhsg&t=520s"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border px-6 py-2.5 text-sm font-medium transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
            >
              Discussion with Abhishek Thakur (4x Kaggle GM)
            </a>
          </div>
        </div>
      </section>
    </>
  );
}
