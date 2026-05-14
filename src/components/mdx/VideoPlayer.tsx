"use client";

interface VideoPlayerProps {
  src: string;
  caption?: string;
}

export function VideoPlayer({ src, caption }: VideoPlayerProps) {
  return (
    <figure className="my-8">
      <video
        src={src}
        controls
        playsInline
        className="w-full rounded-lg"
        style={{ border: "1px solid var(--border)", backgroundColor: "#0F0D0B" }}
      />
      {caption && (
        <figcaption
          className="mt-2 text-center text-sm"
          style={{ color: "var(--text-muted)" }}
        >
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
