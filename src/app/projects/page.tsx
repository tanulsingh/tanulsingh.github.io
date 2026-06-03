import type { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import { ExternalLink, Github as GithubIcon } from "lucide-react";
import type { ProjectData } from "@/types/content";

export const metadata: Metadata = {
  title: "The Workshop",
  description:
    "Code I've shipped — repos, experiments, and projects that taught me something.",
};

interface ProjectEntry extends ProjectData {
  status?: string;
  imageCaption?: string;
}

const projects: ProjectEntry[] = [
  {
    title: "Bits and Surprise — The Two Faces of Entropy",
    description:
      "Entropy and cross-entropy, interpreted two ways — as expected surprise (probabilistic) and as optimal codeword length (information-theoretic). Building Akinator, a Wordle solver, and an LLM 20 Questions agent on top of one shared engine: information gain plus Bayesian belief update.",
    tags: ["Information Theory", "Bayesian Inference", "Kaggle"],
    status: "in progress",
    link: "/projects/two-faces-of-entropy",
    github: "https://github.com/tanulsingh/two-faces-of-entropy",
    image: "/images/projects/two-faces-of-entropy/cover.png",
    imageCaption: "Same equation. Two perspectives.",
  },
  {
    title: "Tackling the Reversal Curse in LLMs — Multi-View Training",
    description:
      "Trained 7B models from scratch on Wikipedia to study the Reversal Curse — the failure where LLMs learn \"A is B\" but can't infer \"B is A.\" Augmenting passages with diverse QA pairs (DeepSeek-R1) generalizes across paraphrases and to un-augmented passages. Inspired by Physics of LLMs.",
    tags: ["LLM Research", "Pre-training", "Reversal Curse"],
    status: "research",
    image: "/images/projects/reversal-curse/cover.png",
    imageCaption: "A → B works. B → A doesn't. Until you train with multiple views.",
  },
];

export default function ProjectsPage() {
  return (
    <div className="lab-page mx-auto max-w-3xl px-6 pb-24 pt-32">
      <header className="mb-12">
        <p
          className="lab-meta mb-2 text-xs uppercase tracking-widest"
          style={{ color: "var(--primary)" }}
        >
          projects/
        </p>
        <h1
          className="mb-3 text-3xl font-bold tracking-tight md:text-4xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Projects
        </h1>
        <p
          className="max-w-2xl text-sm leading-relaxed"
          style={{
            color: "var(--text-secondary)",
            fontFamily: "var(--font-serif)",
          }}
        >
          Repos and experiments worth pointing at. Some are research artifacts
          from work, others are side projects I picked up to understand
          something I couldn&apos;t crack just by reading.
        </p>
      </header>

      <section>
        <h2 className="lab-section-label">Currently on the desk</h2>
        <div>
          {projects.map((project) => {
            const hasImage = !!project.image;
            return (
              <article
                key={project.title}
                className={`lab-row${hasImage ? " has-thumb" : ""}`}
              >
                <div className="lab-date">{project.status || ""}</div>
                {hasImage && (
                  <span className="lab-thumb">
                    <Image
                      src={project.image!}
                      alt={project.title}
                      width={160}
                      height={160}
                    />
                  </span>
                )}
                <div>
                  {project.link ? (
                    <Link
                      href={project.link}
                      {...(project.link.startsWith("http")
                        ? { target: "_blank", rel: "noopener noreferrer" }
                        : {})}
                      className="lab-title hover:opacity-80"
                      style={{ display: "block" }}
                    >
                      {project.title}
                    </Link>
                  ) : (
                    <div className="lab-title">{project.title}</div>
                  )}
                  {hasImage && project.imageCaption ? (
                    <p className="lab-caption">{project.imageCaption}</p>
                  ) : (
                    <p className="lab-blurb">{project.description}</p>
                  )}
                  <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
                    {project.tags.map((tag) => (
                      <span
                        key={tag}
                        className="font-mono text-xs"
                        style={{ color: "var(--text-muted)" }}
                      >
                        · {tag}
                      </span>
                    ))}
                  </div>
                  {(project.link || project.github) && (
                    <div className="mt-3 flex flex-wrap gap-5">
                      {project.link && (
                        <Link
                          href={project.link}
                          {...(project.link.startsWith("http")
                            ? { target: "_blank", rel: "noopener noreferrer" }
                            : {})}
                          className="flex items-center gap-1.5 font-mono text-xs hover:opacity-80"
                          style={{ color: "var(--primary)" }}
                        >
                          <ExternalLink size={12} />
                          {project.link.startsWith("http")
                            ? "view"
                            : "read more"}
                        </Link>
                      )}
                      {project.github && (
                        <Link
                          href={project.github}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-1.5 font-mono text-xs hover:opacity-80"
                          style={{ color: "var(--text-muted)" }}
                        >
                          <GithubIcon size={12} />
                          source
                        </Link>
                      )}
                    </div>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
