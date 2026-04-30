"use client";

import { motion } from "motion/react";
import Link from "next/link";
import Image from "next/image";
import { ArrowDown } from "lucide-react";

export function HeroSection() {
  return (
    <section className="relative flex min-h-screen items-center overflow-hidden px-6">
      <div className="relative mx-auto w-full max-w-4xl">
        {/* Photo + Name */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="mb-6 flex items-center gap-5"
        >
          <div className="relative">
            <div
              className="absolute -inset-1 rounded-full opacity-30 blur-md"
              style={{
                background: "linear-gradient(135deg, var(--gradient-from), var(--gradient-to))",
              }}
            />
            <Image
              src="/images/profile/402A2908.JPG"
              alt="Tanul Singh"
              width={64}
              height={64}
              className="relative rounded-full object-cover"
              style={{ border: "2px solid var(--border)" }}
              priority
            />
          </div>
          <div>
            <h1
              className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Tanul Singh
            </h1>
            <p className="font-mono text-xs" style={{ color: "var(--text-muted)" }}>
              ML Engineer &middot; 5+ years in NLP &amp; LLMs
            </p>
          </div>
        </motion.div>

        {/* ML wordplay tagline */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-4 font-mono text-sm md:text-base"
          style={{ color: "var(--primary)" }}
        >
          # trained from scratch. no pre-trained weights.
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.35 }}
          className="mb-12 max-w-xl leading-relaxed"
          style={{ color: "var(--text-secondary)" }}
        >
          Initialized with a Mechanical Engineering degree, pre-trained on
          curiosity, and fine-tuned by an unreasonable number of late nights
          — 5+ years of gradient descent through NLP, LLMs, and generative AI.
          Currently inference-serving at{" "}
          <span className="font-semibold" style={{ color: "var(--text)" }}>
            Apple
          </span>
          . I write about ML here so you don&apos;t have to train from scratch too.
        </motion.p>

        {/* Subtle prompt to explore */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.55 }}
          className="flex flex-wrap items-center gap-6"
        >
          <Link
            href="/blog"
            className="font-mono text-sm transition-colors hover:opacity-80"
            style={{ color: "var(--text-secondary)" }}
          >
            &rarr; read the blog
          </Link>
          <Link
            href="#training-loop"
            className="font-mono text-sm transition-colors hover:opacity-80"
            style={{ color: "var(--text-secondary)" }}
          >
            &rarr; see my training curve
          </Link>
          <Link
            href="/about"
            className="font-mono text-sm transition-colors hover:opacity-80"
            style={{ color: "var(--text-secondary)" }}
          >
            &rarr; see my full forward pass
          </Link>
        </motion.div>

        {/* Scroll hint */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.4 }}
          transition={{ duration: 0.6, delay: 1.2 }}
          className="absolute hidden md:block"
          style={{ bottom: "-120px", left: 0 }}
        >
          <ArrowDown size={16} className="animate-bounce" style={{ color: "var(--text-muted)" }} />
        </motion.div>
      </div>
    </section>
  );
}
