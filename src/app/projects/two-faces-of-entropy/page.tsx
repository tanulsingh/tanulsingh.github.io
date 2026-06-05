import type { Metadata } from "next";
import Link from "next/link";
import { ExternalLink, Github } from "lucide-react";
import { getNotesByProject } from "@/lib/content";
import { formatDate } from "@/lib/utils";

const GITHUB_URL = "https://github.com/tanulsingh/two-faces-of-entropy";
const BLOG_URL = "/blog/llm-pretraining";
const PROJECT_SLUG = "two-faces-of-entropy";

export const metadata: Metadata = {
  title: "Bits and Surprise — The Two Faces of Entropy",
  description:
    "Two interpretations of entropy and cross-entropy — probabilistic (expected surprise) and information-theoretic (compression) — paired with applied projects that show what each view buys you.",
};

export default function TwoFacesOfEntropyPage() {
  const notes = getNotesByProject(PROJECT_SLUG).filter((n) => !n.frontmatter.draft);

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
          <Link
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-opacity hover:opacity-90"
            style={{
              backgroundColor: "var(--primary)",
              color: "var(--bg)",
            }}
          >
            <Github size={14} />
            View on GitHub
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

        {/* The Project */}
        <h2>The Project</h2>

        <h3>Akinator — Inspired from Wordle Solution but Easier</h3>

        <p>
          If you&apos;ve watched Grant Sanderson&apos;s{" "}
          <a
            href="https://www.youtube.com/watch?v=v68zYyaEmEA"
            target="_blank"
            rel="noopener noreferrer"
          >
            Wordle video
          </a>
          , you&apos;ve already seen the canonical demo of information theory in
          action — entropy picking the optimal guess at every step, the solver
          narrowing 13,000 candidate words down to one. He even walks through
          the subtle moments — like when the solver gets down to two candidate
          words and has no way to break the tie except guessing one and hoping,
          then upgrades the solver with word-frequency priors so it learns to
          prefer the more common candidate. If you haven&apos;t watched it yet,
          go do that first. It&apos;s twenty minutes and it&apos;ll change how
          you think about guessing games forever.
        </p>

        <p>
          I tried implementing it myself. And the <em>theory</em> clicked
          instantly — but the <em>implementation</em> didn&apos;t.
          Wordle&apos;s feedback isn&apos;t binary; every guess produces one of
          243 possible colour patterns (5 squares × 3 colours each). So every
          entropy calculation is a 243-bucket sum, every belief update is a
          243-pattern filter, and the algorithm hides under the combinatorial
          machinery. You can verify it works statistically, but you can&apos;t{" "}
          <em>feel</em> it work — the steps are too dense to follow by hand,
          the bookkeeping too heavy to debug by inspection.
        </p>

        <p>
          So instead of redoing Wordle (Grant already does it definitively), I
          built{" "}
          <strong>
            <a
              href={`${GITHUB_URL}/tree/main/akinator`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Akinator
            </a>
          </strong>{" "}
          — same engine on a smaller, predictable domain. 46 animals, 18 binary
          features, yes/no questions only. The whole entropy table fits on a
          screen, you can compute any step on a napkin, and you can{" "}
          <em>see</em> why the solver picks what features at each step
        </p>

        <p>
          I deliberately built in the same arc as Grant&apos;s video, just
          slower and more visible at every step:
        </p>

        <ul>
          <li>
            <strong>Start with uniform priors</strong> — every animal equally
            likely. The solver works, the math holds, ~5.6 questions average.
            But two pairs of animals (Horse/Cow, Pig/Rabbit) have{" "}
            <em>identical feature vectors</em>, and the solver can&apos;t break
            the tie no matter what it asks — the same dead-end Grant hits with
            two-word ambiguity in Wordle.
          </li>
          <li>
            <strong>Fix the dataset.</strong> Add two features that target the
            indistinguishability directly. The dead-ends vanish.
          </li>
          <li>
            <strong>Add non-uniform priors</strong> — Just like in Wordle not
            all the candidate words can be answer and some occur more than the
            other, animals also differ in popularity and common animals might
            be the answer in real life than rarer ones. Lion gets a higher
            prior than Platypus, because that&apos;s what a real player would
            pick. Watch the solver&apos;s strategy shift from{" "}
            <em>&ldquo;split the candidate pool&rdquo;</em> to{" "}
            <em>&ldquo;split the probability mass.&rdquo;</em> Common animals
            get found in 4 questions instead of 5. The same move Grant makes
            when he adds word-frequency priors from a Google dataset.
          </li>
        </ul>

        <p>
          Three experiments, written up in{" "}
          <a
            href={`${GITHUB_URL}/blob/main/akinator/README.md`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <code>akinator/README.md</code>
          </a>
          , each with the numbers and the surprises. The point is not the bot —
          it&apos;s that <em>every</em> concept in information theory has a
          small, predictable version you can verify by hand before you trust it
          at scale.
        </p>

        <p>
          <strong>If you want the deeper learning loop</strong>: watch
          Grant&apos;s video, then try implementing Wordle yourself. When the
          243-pattern machinery slows you down (it will), come read through
          Akinator. Both projects does the same thing; one just hides nothing.
        </p>

        <h3>
          What&apos;s next — LLM 20 Questions <em>(coming soon)</em>
        </h3>

        <p>
          The natural extension is to swap the deterministic answerer for an
          LLM — noisy, sometimes wrong, candidate space unbounded. The same
          engine should still work but with Bayesian belief update instead of
          hard filtering. That&apos;s the bridge from Akinator (clean) to
          real-world systems (messy). Building this next as an agent for
          Kaggle&apos;s{" "}
          <a
            href="https://www.kaggle.com/competitions/llm-20-questions"
            target="_blank"
            rel="noopener noreferrer"
          >
            LLM 20 Questions
          </a>{" "}
          competition.
        </p>

        {/* Learning notes — at the end, Olah-style */}
        <h2>Learning notes</h2>

        <p>
          A handful of intuitions that clicked while I was building this. Each
          one is a question I had and what I figured out, written down so
          I&apos;d remember the moment things stopped feeling abstract.
        </p>

        {notes.length > 0 ? (
          <ul style={{ listStyle: "none", paddingLeft: 0 }}>
            {notes.map((note) => (
              <li key={note.slug} style={{ marginBottom: "0.5rem" }}>
                <Link href={`/notes/${note.slug}`}>
                  {note.frontmatter.title}
                </Link>{" "}
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.7rem",
                    color: "var(--text-muted)",
                    marginLeft: "0.5rem",
                  }}
                >
                  {formatDate(note.frontmatter.date)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p style={{ fontStyle: "italic", color: "var(--text-muted)" }}>
            Notes coming soon.
          </p>
        )}
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
