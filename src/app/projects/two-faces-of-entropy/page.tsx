import type { Metadata } from "next";
import Link from "next/link";
import { ExternalLink, Github } from "lucide-react";

const GITHUB_URL = "https://github.com/tanulsingh/two-faces-of-entropy";
const BLOG_URL = "/blog/llm-pretraining";

export const metadata: Metadata = {
  title: "Bits and Surprise — The Two Faces of Entropy",
  description:
    "Two interpretations of entropy and cross-entropy — probabilistic (expected surprise) and information-theoretic (compression) — paired with applied projects that show what each view buys you.",
};

export default function TwoFacesOfEntropyPage() {
  return (
    <article className="mx-auto max-w-3xl px-6 pb-24 pt-32">
      {/* Header */}
      <header className="mb-12">
        <Link
          href="/projects"
          className="mb-4 inline-block font-mono text-xs hover:opacity-80"
          style={{ color: "var(--primary)" }}
        >
          ← projects/
        </Link>

        <h1
          className="mb-3 text-4xl font-bold tracking-tight md:text-5xl"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Bits and Surprise
        </h1>
        <p
          className="mb-6 text-lg italic"
          style={{ color: "var(--text-secondary)", fontFamily: "var(--font-serif)" }}
        >
          The Two Faces of Entropy
        </p>

        <div className="mb-6 flex flex-wrap items-center gap-4">
          <span
            className="rounded px-2 py-0.5 font-mono text-xs"
            style={{
              backgroundColor: "var(--tag-bg)",
              color: "var(--tag-text)",
            }}
          >
            In Progress
          </span>
          <Link
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-sm font-medium hover:opacity-80"
            style={{ color: "var(--text-muted)" }}
          >
            <Github size={14} />
            Source
          </Link>
        </div>

        <p
          className="text-base"
          style={{ color: "var(--text-secondary)" }}
        >
          Information theory, two ways — and what happens when you stop nodding
          politely at it and actually use it.
        </p>
      </header>

      <div className="prose prose-lg prose-paper max-w-none dark:prose-invert">
        {/* The story */}
        <h2>The story</h2>

        <p>
          As an ML engineer, I used to reach for <code>cross_entropy_loss</code> the
          way I trust the default Adam hyperparameters — full confidence, zero
          scrutiny. I knew the formula, I knew how it punishes wrong predictions and
          rewards confident-correct ones. The number goes down, the model gets
          better, what&apos;s not to like. But I never <em>really</em> understood it.
          I didn&apos;t even know why I was choosing it in the first place when there
          are so many other losses out there doing apparently the same job (how is
          this one any different?). What me and cross-entropy had was a very
          surface-level relationship, and I wanted something more intimate, you know
          — because I&apos;m stuck with this thing for life :-p
        </p>

        <p>For that I needed to know it better and get some answers:</p>

        <ul>
          <li>
            What does a loss value of <code>3.34</code> actually <em>mean</em>?
          </li>
          <li>
            How do I interpret cross-entropy <em>physically</em> — not as a number on
            a tensorboard, but as a quantity in the world?
          </li>
          <li>Can I observe cross-entropy outside of an ML training loop?</li>
        </ul>

        <p>
          So I went looking. The first thing that gave me real ground was{" "}
          <strong>
            Chapter 5.5 (&ldquo;Maximum Likelihood Estimation&rdquo;) and Chapter
            6.2.1.1 (&ldquo;Learning Conditional Distributions with Maximum
            Likelihood&rdquo;) of{" "}
            <a
              href="https://www.deeplearningbook.org/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Goodfellow, Bengio &amp; Courville&apos;s <em>Deep Learning</em>
            </a>
          </strong>
          . Those two chapters walked me through the whole chain: minimising KL
          divergence between the empirical data distribution and the model
          distribution is mathematically the same as maximising log-likelihood, which
          collapses neatly into cross-entropy. Reading them back-to-back, &ldquo;why
          cross-entropy?&rdquo; stopped being a mystery — it&apos;s just what you get
          when you take MLE seriously. I felt like I finally <em>knew</em>{" "}
          cross-entropy, but our friend here is way more mysterious than I thought.
        </p>

        <p>
          The math part was covered but I still wanted the bigger picture —
          surprisal, entropy, KL divergence — not just CE sitting in isolation.
          That&apos;s when I found{" "}
          <a
            href="https://www.youtube.com/watch?v=KHVR587oW8I"
            target="_blank"
            rel="noopener noreferrer"
          >
            Artem Kirsanov&apos;s &ldquo;The Key Equation Behind Probability&rdquo;
          </a>{" "}
          which builds the entire stack from one simple intuition (
          <em>how surprised would you be?</em>) all the way up through cross-entropy
          and KL divergence. Cleanest derivation of the whole family I&apos;ve seen,
          and the kind of video you want to re-watch a week later just to feel smart
          again. This is the point where I started falling for my dearest Cross
          Entropy.
        </p>

        <p>
          I now felt very comfortable with the <strong>probabilistic view</strong>.
          Hubris achieved, life was good. Then last week, while I was starting to
          write the first post of my new blog series,{" "}
          <Link href={BLOG_URL}>The Loss Landscape of LLM Training</Link>, I tripped
          over the line <strong>&ldquo;language models are compressors&rdquo;</strong>.
          I read it five times trying to figure out which word was the typo, then
          went looking for an explanation — only to find out that cross-entropy is{" "}
          <em>literally</em> the number of bits per symbol your model would use to
          compress the data. Same formula. Completely different universe. I sat with
          that one for a while, had a laugh or two, you sly little fox, how much more
          efforts do you wanna take?
        </p>

        <p>
          That was the second half: the{" "}
          <strong>information-theoretic view</strong>. Chris Olah&apos;s{" "}
          <a
            href="https://colah.github.io/posts/2015-09-Visual-Information/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Visual Information Theory
          </a>{" "}
          made the connection sing, and Grant Sanderson&apos;s{" "}
          <a
            href="https://www.youtube.com/watch?v=v68zYyaEmEA"
            target="_blank"
            rel="noopener noreferrer"
          >
            Solving Wordle using Information Theory
          </a>{" "}
          made it feel like a tool I could actually pick up and use rather than just
          admire from a distance.
        </p>

        <p>
          You see the beauty of this thing? How should one not fall in Love? Two
          completely different stories about the exact same equation. Once you see
          both, you can never go back to seeing only one. This repo is what came out
          of chasing that thread — instead of writing yet another blog post
          explaining entropy (the resources below already do it much better than I
          ever could), I wanted to actually <em>apply</em> it. Build real things
          where information theory is the engine, and see whether the intuition
          holds up when it has to drive code that works.
        </p>

        <p>Spoiler: it does. Mostly.</p>

        {/* Where to start */}
        <h2>Where to start</h2>

        <p>
          If like me you also want a deeper relationship with Cross Entropy, Entropy
          and Uncertainty in general, here&apos;s what I would do, read these in
          order.
        </p>

        <h3>The probabilistic view</h3>

        <ol>
          <li>
            <strong>
              <a
                href="https://www.deeplearningbook.org/"
                target="_blank"
                rel="noopener noreferrer"
              >
                <em>Deep Learning</em> — Chapter 5.5 + 6.2.1.1
              </a>
            </strong>{" "}
            (Goodfellow, Bengio, Courville). The formal derivation — KL divergence ↔
            negative log-likelihood ↔ cross-entropy, all in a few pages. If you want
            the proof, this is the proof.
          </li>
          <li>
            <strong>
              <a
                href="https://www.youtube.com/watch?v=KHVR587oW8I"
                target="_blank"
                rel="noopener noreferrer"
              >
                The Key Equation Behind Probability
              </a>
            </strong>{" "}
            — Artem Kirsanov. The intuitive companion to the math. Builds surprisal
            → entropy → cross-entropy → KL divergence from a single intuition (
            <em>how surprised would you be?</em>). Watch this either before or after
            the book chapters — they reinforce each other beautifully.
          </li>
        </ol>

        <h3>The information-theoretic view</h3>

        <ol start={3}>
          <li>
            <strong>
              <a
                href="https://colah.github.io/posts/2015-09-Visual-Information/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Visual Information Theory
              </a>
            </strong>{" "}
            — Chris Olah. The single best visual introduction to all of this.
            Reframes the same formulas through codes and bits — entropy as
            &ldquo;optimal codeword length&rdquo;, cross-entropy as &ldquo;what you
            pay when you use the wrong codebook.&rdquo; Once you&apos;ve read this,
            the compression view will feel obvious in hindsight (it really
            isn&apos;t).
          </li>
          <li>
            <strong>
              <a
                href="https://www.youtube.com/watch?v=v68zYyaEmEA"
                target="_blank"
                rel="noopener noreferrer"
              >
                Solving Wordle using Information Theory
              </a>
            </strong>{" "}
            — Grant Sanderson (3Blue1Brown). Turns &ldquo;information gain&rdquo;
            into something you can <em>feel</em> by watching it solve a game in real
            time. Also the direct inspiration for two of the projects in this repo.
          </li>
        </ol>

        <p>
          After those four, you&apos;ll feel the love I am feeling for our dearest
          Cross Entropy and Uncertainty.
        </p>

        {/* The projects */}
        <h2>The projects</h2>

        <p>
          Each project applies the same engine (entropy, information gain, Bayesian
          update) to a different problem of increasing nastiness.
        </p>

        <ul>
          <li>
            <strong>
              <a
                href={`${GITHUB_URL}/tree/main/akinator`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <code>akinator/</code>
              </a>
            </strong>{" "}
            — the warm-up. A 20-questions-style solver over a small hand-authored
            celebrity table. The cleanest possible demonstration of &ldquo;pick the
            question with maximum expected information gain.&rdquo; If you can build
            this, you understand the algorithm.
          </li>
          <li>
            <strong>
              <a
                href={`${GITHUB_URL}/tree/main/wordle`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <code>wordle/</code>
              </a>
            </strong>{" "}
            — the canonical example, popularised by 3B1B. An optimal Wordle solver
            with frequency-based priors and the full 243-pattern entropy machinery.
            Same algorithm as Akinator but with significantly more engineering
            attached.
          </li>
          <li>
            <strong>
              <a
                href={`${GITHUB_URL}/tree/main/llm-20-questions`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <code>llm-20-questions/</code>
              </a>
            </strong>{" "}
            — the capstone. An agent for Kaggle&apos;s{" "}
            <a
              href="https://www.kaggle.com/competitions/llm-20-questions"
              target="_blank"
              rel="noopener noreferrer"
            >
              LLM 20 Questions
            </a>{" "}
            competition that uses the same engine but with an LLM as a very noisy
            oracle. Everything that <em>can</em> go wrong here does, which is kind
            of the whole point.
          </li>
        </ul>

        <p>
          More to come as I find new things to chase. Each project has its own
          README with the algorithm, the result, and whatever surprised me along the
          way.
        </p>

        {/* Why this repo exists */}
        <h2>Why this repo exists</h2>

        <p>
          This repo exists to admire the beauty of Cross Entropy, Entropy and
          Uncertainty and how elegantly they summarise this feeling in mathematical
          form. We get handed information-theoretic tools in ML as if they&apos;re
          just formulas to plug in. They&apos;re not — they&apos;re a whole way of
          thinking about uncertainty, observation, and belief update that goes way
          beyond loss functions. The best way I know to actually internalise
          something like that is to build things where the formulas have to carry
          real weight. If a bot can&apos;t guess &ldquo;Beyoncé&rdquo; in seven
          questions, the formula failed <em>me</em> — not the other way around.
        </p>

        <p>
          If you came here just for the code: it&apos;s all in the project folders.
          If you came to chase the same thread I did: start with the four links
          above, then poke around.
        </p>
      </div>

      {/* Footer with link back */}
      <div
        className="mt-16 flex items-center justify-between border-t pt-8"
        style={{ borderColor: "var(--border)" }}
      >
        <Link
          href="/projects"
          className="text-sm font-medium hover:opacity-80"
          style={{ color: "var(--primary)" }}
        >
          ← All projects
        </Link>
        <Link
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-sm font-medium hover:opacity-80"
          style={{ color: "var(--text-muted)" }}
        >
          <Github size={14} />
          View on GitHub
          <ExternalLink size={12} />
        </Link>
      </div>
    </article>
  );
}
