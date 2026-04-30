import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-6">
      <h1 className="mb-2 text-7xl font-bold gradient-text">404</h1>
      <p className="mb-8 text-lg" style={{ color: "var(--text-secondary)" }}>
        This page doesn&apos;t exist yet.
      </p>
      <Link
        href="/"
        className="rounded-full px-8 py-3 text-sm font-semibold text-white"
        style={{ backgroundColor: "var(--primary)" }}
      >
        Go Home
      </Link>
    </div>
  );
}
