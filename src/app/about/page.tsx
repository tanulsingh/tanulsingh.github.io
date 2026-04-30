import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { Github, Linkedin, Twitter, Mail, ExternalLink } from "lucide-react";

export const metadata: Metadata = {
  title: "About",
  description:
    "From a small town in Uttar Pradesh to Senior ML Engineer at Apple — the full story of a self-taught AI practitioner.",
};

const courses = [
  { name: "Gilbert Strang's Linear Algebra", url: "https://www.youtube.com/playlist?list=PL221E2BBF13BECF6C" },
  { name: "Stanford CS109: Probability for CS", url: "https://www.youtube.com/playlist?list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg" },
  { name: "Stanford CS231n: CNNs for Visual Recognition", url: "https://cs231n.github.io/" },
  { name: "Stanford CS224n: NLP with Deep Learning", url: "https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4" },
  { name: "Stanford CS229: Machine Learning", url: "https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU" },
  { name: "Berkeley Deep Reinforcement Learning", url: "https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps" },
];

const achievements = [
  {
    title: "Kaggle Days Championship Winner",
    description: "Won Regionals, represented India in the finals in Barcelona, Spain.",
    url: "https://www.linkedin.com/feed/update/urn:li:activity:6909399161356320768/",
  },
  {
    title: "Kaggle Competitions Master",
    description: "3 gold and 10 silver medals across 30+ competitions.",
    url: "https://www.kaggle.com/tanulsingh077",
  },
  {
    title: "Kaggle Notebooks Grandmaster",
    description: "16 gold medals, showcasing solutions across multiple verticals.",
    url: "https://www.kaggle.com/tanulsingh077",
  },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      {/* Hero */}
      <div className="mb-16 flex flex-col items-center gap-8 md:flex-row md:items-start">
        <div className="relative shrink-0">
          <div
            className="absolute -inset-1.5 rounded-2xl opacity-50 blur-lg"
            style={{
              background: "linear-gradient(135deg, var(--gradient-from), var(--gradient-to))",
            }}
          />
          <Image
            src="/images/profile/20230929_125148.jpg"
            alt="Tanul Singh"
            width={200}
            height={200}
            className="relative rounded-2xl object-cover"
            style={{ border: "2px solid var(--border)" }}
            priority
          />
        </div>
        <div>
          <p
            className="mb-2 text-sm font-semibold uppercase tracking-[0.15em]"
            style={{ color: "var(--primary)" }}
          >
            About Me
          </p>
          <h1 className="mb-4 text-4xl font-bold tracking-tight">
            Tanul Singh
          </h1>
          <p
            className="mb-2 text-lg leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            Senior Machine Learning Engineer at{" "}
            <strong style={{ color: "var(--text)" }}>Apple</strong>.
            Kaggle Grandmaster. US Patent holder. Published researcher.
          </p>
          <p
            className="mb-6 leading-relaxed"
            style={{ color: "var(--text-muted)" }}
          >
            Self-taught from Mechanical Engineering — no CS degree, no fancy pedigree.
            Just an unshakeable belief that the drive to learn is more powerful
            than any credential.
          </p>
          <div className="flex gap-3">
            {[
              { icon: <Github size={18} />, url: "https://github.com/tanulsingh" },
              { icon: <Linkedin size={18} />, url: "https://www.linkedin.com/in/tanul-singh/" },
              { icon: <Twitter size={18} />, url: "https://x.com/singh_tanul" },
              { icon: <Mail size={18} />, url: "mailto:tanulsingh0077@gmail.com" },
            ].map((s) => (
              <Link
                key={s.url}
                href={s.url}
                target={s.url.startsWith("mailto") ? undefined : "_blank"}
                rel={s.url.startsWith("mailto") ? undefined : "noopener noreferrer"}
                className="flex h-10 w-10 items-center justify-center rounded-lg border transition-colors hover:opacity-80"
                style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}
              >
                {s.icon}
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* The Story */}
      <section className="mb-16">
        <h2 className="mb-6 text-2xl font-bold">The Story</h2>
        <div className="space-y-4 leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          <p>
            I come from a very small town in Uttar Pradesh, India. Growing up, the only
            path to success anyone knew was: study hard, clear IIT, get a job at an MNC.
            I studied hard through my 12th, but I didn&apos;t clear IIT. I ended up at the best
            college in my state, studying Mechanical Engineering — a branch I got because of a bad rank.
          </p>
          <p>
            But my heart was always in Computer Science. In my second year (2016), a senior
            mentioned Machine Learning. I found Andrew Ng&apos;s course, tried the digit recognizer,
            and was instantly hooked. My father was a maths teacher — seeing how we can express
            the intuitive things we experience in life through mathematics just amazed me.
          </p>
          <p style={{ color: "var(--text)" }}>
            <strong>I lost my father in my second year of college.</strong> With no earning member
            in the family, everything fell on me. But I was so hooked on ML that I didn&apos;t stop.
            I went to college during the day, tutored kids to raise funds, and studied Python and
            ML deep into the night.
          </p>
          <p>
            Being in India, I didn&apos;t have access to good teachers or people who were doing ML
            work at the time. But I found Kaggle — I learned from notebooks shared by the best
            in the world, people I&apos;d never meet. After two years of this routine, I became a
            Kaggle Competitions Expert and secured a Data Science job — the first person from
            my college to do so.
          </p>
          <p>
            In September 2020, I made a promise to myself: <em>someday, I will work as a
            Research Scientist at a world-class lab, leading the frontier of AI.</em> Every step
            since then has been toward that goal — Kaggle Grandmaster, Stanford courses, published
            research, a US patent, building production ML at LevelAI, and now working at Apple
            where I collaborate with the ML Research team.
          </p>
        </div>
      </section>

      {/* What I Do at Apple */}
      <section className="mb-16">
        <h2 className="mb-6 text-2xl font-bold">What I Do at Apple</h2>
        <div className="space-y-4 leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          <p>
            My work at Apple sits at the intersection of research and applied AI —
            roughly half my time goes into ML research problems and the other half
            into building production systems that ship to millions of users.
          </p>
          <p>
            On the research side, I collaborate with Apple&apos;s ML Research team to
            investigate open problems in LLM generalization, knowledge representation,
            and model robustness. On the applied side, I build and deploy large-scale
            NLP and generative AI systems — from multi-agentic architectures to
            safety and guardrail frameworks.
          </p>
          <p>
            It&apos;s the kind of role I&apos;d been working toward since 2020 —
            one foot in research, one foot in production, and both feet in problems
            that actually matter at scale.
          </p>
        </div>
      </section>

      {/* Patents & Publications */}
      <section className="mb-16">
        <h2 className="mb-6 text-2xl font-bold">Patents &amp; Publications</h2>
        <div className="space-y-4">
          <Link
            href="https://pubchem.ncbi.nlm.nih.gov/patent/US-12361933-B1"
            target="_blank"
            rel="noopener noreferrer"
            className="group block rounded-xl border p-5 transition-colors hover:opacity-90"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
          >
            <div className="flex items-start justify-between gap-2">
              <h3 className="font-semibold">US Patent — Dynamic Intent Detection</h3>
              <ExternalLink size={16} className="mt-1 shrink-0" style={{ color: "var(--text-muted)" }} />
            </div>
            <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
              A real-time intent detection system for customer-agent interactions, adaptable to
              client modifications without retraining — surpassing traditional static models.
            </p>
          </Link>
          <Link
            href="https://www.researchgate.net/publication/359051366_GWNET_Detecting_Gravitational_Waves_using_Hierarchical_and_Residual_Learning_based_1D_CNNs"
            target="_blank"
            rel="noopener noreferrer"
            className="group block rounded-xl border p-5 transition-colors hover:opacity-90"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
          >
            <div className="flex items-start justify-between gap-2">
              <h3 className="font-semibold">
                GWNET: Detecting Gravitational Waves using 1D CNNs
              </h3>
              <ExternalLink size={16} className="mt-1 shrink-0" style={{ color: "var(--text-muted)" }} />
            </div>
            <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
              Gold-winning Kaggle solution. End-to-end architecture for detecting gravitational
              waves from raw interferometer time-series data.
            </p>
          </Link>
        </div>
      </section>

      {/* Kaggle */}
      <section className="mb-16">
        <h2 className="mb-6 text-2xl font-bold">Kaggle Achievements</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {achievements.map((a) => (
            <Link
              key={a.title}
              href={a.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group block rounded-xl border p-5 transition-colors hover:opacity-90"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
            >
              <h3 className="mb-1 font-semibold">{a.title}</h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                {a.description}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Self-Teaching */}
      <section className="mb-16">
        <h2 className="mb-6 text-2xl font-bold">How I Learned</h2>
        <p className="mb-4" style={{ color: "var(--text-secondary)" }}>
          No degree in CS or AI. These courses, plus thousands of hours on Kaggle
          and reading papers, are what built my foundation:
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {courses.map((c) => (
            <Link
              key={c.name}
              href={c.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-center gap-2 rounded-lg border px-4 py-3 text-sm transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
            >
              <span style={{ color: "var(--text-secondary)" }}>{c.name}</span>
              <ExternalLink size={12} className="shrink-0" style={{ color: "var(--text-muted)" }} />
            </Link>
          ))}
        </div>
      </section>

      {/* Conversations */}
      <section>
        <h2 className="mb-6 text-2xl font-bold">Featured Conversations</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <Link
            href="https://www.youtube.com/watch?v=t-kiCW0jdLg&t=490s"
            target="_blank"
            rel="noopener noreferrer"
            className="group block rounded-xl border p-5 transition-colors hover:opacity-90"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
          >
            <h3 className="mb-1 font-semibold">Discussion with Dhruv</h3>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              Founder of CodeBasics
            </p>
          </Link>
          <Link
            href="https://www.youtube.com/watch?v=ujGLgn3fhsg&t=520s"
            target="_blank"
            rel="noopener noreferrer"
            className="group block rounded-xl border p-5 transition-colors hover:opacity-90"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
          >
            <h3 className="mb-1 font-semibold">Discussion with Abhishek Thakur</h3>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              4x Kaggle Grandmaster
            </p>
          </Link>
        </div>
      </section>
    </div>
  );
}
