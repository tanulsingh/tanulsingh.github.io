import type { Metadata } from "next";
import Link from "next/link";
import { ExternalLink, Github } from "lucide-react";
import type { ProjectData } from "@/types/content";

export const metadata: Metadata = {
  title: "Projects",
  description: "ML projects, open source contributions, and Kaggle competition solutions.",
};

const projects: ProjectData[] = [
  {
    title: "ML Research & Applied AI at Apple",
    description:
      "Split between ML research (LLM generalization, knowledge representation, model robustness) and production AI systems (multi-agentic architectures, generative AI, safety frameworks) serving millions of users.",
    tags: ["LLM Research", "Applied AI", "NLP", "Apple"],
  },
  {
    title: "GWNET — Gravitational Wave Detection",
    description:
      "End-to-end 1D CNN architecture for detecting gravitational waves directly from raw interferometer time-series data. Gold medal winning Kaggle solution and published research paper.",
    tags: ["PyTorch", "1D CNN", "Signal Processing", "Kaggle"],
    link: "https://www.researchgate.net/publication/359051366_GWNET_Detecting_Gravitational_Waves_using_Hierarchical_and_Residual_Learning_based_1D_CNNs",
    github: "https://github.com/tanulsingh",
  },
  {
    title: "Dynamic Intent Detection System (US Patent)",
    description:
      "US-patented system for real-time intent detection from customer-agent interactions, adaptable to client modifications without retraining. Surpasses traditional static models.",
    tags: ["NLP", "Real-time ML", "Patent", "LevelAI"],
  },
  {
    title: "LLM-based Contact Center Autoscoring",
    description:
      "Fine-tuned Mixtral-7B to autoscore contact center agents by answering diverse rubric questions in zero-shot. Scaled across all clients at LevelAI.",
    tags: ["LLMs", "Fine-tuning", "Mixtral", "LevelAI"],
  },
];

export default function ProjectsPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <div className="mb-12">
        <p className="mb-1 font-mono text-sm" style={{ color: "var(--primary)" }}>
          projects/
        </p>
        <h1
          className="mb-3 text-4xl font-bold tracking-tight md:text-5xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Things I&apos;ve Built
        </h1>
        <p
          className="max-w-2xl text-lg"
          style={{ color: "var(--text-secondary)" }}
        >
          From Kaggle competition solutions to production ML systems serving
          millions of interactions.
        </p>
      </div>

      <div className="grid gap-6">
        {projects.map((project) => (
          <div
            key={project.title}
            className="notebook-cell p-6"
          >
            <div className="mb-3 flex flex-wrap gap-2">
              {project.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded px-2 py-0.5 font-mono text-xs"
                  style={{
                    backgroundColor: "var(--tag-bg)",
                    color: "var(--tag-text)",
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
            <h3 className="mb-2 text-xl font-bold">{project.title}</h3>
            <p
              className="mb-4 text-sm leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              {project.description}
            </p>
            {(project.link || project.github) && (
              <div className="flex gap-4">
                {project.link && (
                  <Link
                    href={project.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-sm font-medium hover:opacity-80"
                    style={{ color: "var(--primary)" }}
                  >
                    <ExternalLink size={14} />
                    View Project
                  </Link>
                )}
                {project.github && (
                  <Link
                    href={project.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-sm font-medium hover:opacity-80"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <Github size={14} />
                    Source
                  </Link>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
