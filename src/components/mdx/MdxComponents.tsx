import type { MDXComponents } from "mdx/types";
import { Callout } from "./Callout";
import { BPEVisualizer } from "./BPEVisualizer";
import { UnigramVisualizer } from "./UnigramVisualizer";
import { SentencePieceDemo } from "./SentencePieceDemo";
import { PEVisualizer } from "./PEVisualizer";
import { RoPEVisualizer } from "./RoPEVisualizer";

export const mdxComponents: MDXComponents = {
  Callout,
  BPEVisualizer,
  UnigramVisualizer,
  SentencePieceDemo,
  PEVisualizer,
  RoPEVisualizer,

  img: (props) => (
    <figure className="my-8">
      <img
        {...props}
        className="w-full rounded-lg"
        style={{ border: "1px solid var(--border)" }}
        loading="lazy"
      />
      {props.alt && (
        <figcaption
          className="mt-2 text-center text-sm"
          style={{ color: "var(--text-muted)" }}
        >
          {props.alt}
        </figcaption>
      )}
    </figure>
  ),

  a: (props) => {
    const isExternal =
      props.href?.startsWith("http") || props.href?.startsWith("mailto");
    return (
      <a
        {...props}
        style={{ color: "var(--primary)" }}
        className="font-medium underline underline-offset-2 hover:opacity-80"
        {...(isExternal && { target: "_blank", rel: "noopener noreferrer" })}
      />
    );
  },

  blockquote: (props) => (
    <blockquote
      {...props}
      className="my-6 border-l-2 pl-4 italic"
      style={{
        borderColor: "var(--primary)",
        color: "var(--text-secondary)",
      }}
    />
  ),

  table: (props) => (
    <div
      className="my-6 overflow-x-auto rounded-lg border"
      style={{ borderColor: "var(--border)" }}
    >
      <table {...props} className="w-full text-sm" />
    </div>
  ),

  th: (props) => (
    <th
      {...props}
      className="border-b px-4 py-2 text-left font-semibold"
      style={{
        borderColor: "var(--border)",
        backgroundColor: "var(--bg-elevated)",
      }}
    />
  ),

  td: (props) => (
    <td
      {...props}
      className="border-b px-4 py-2"
      style={{ borderColor: "var(--border)" }}
    />
  ),

  hr: () => <hr className="my-8" style={{ borderColor: "var(--border)" }} />,
};
