import Link from "next/link";
import { Github, Linkedin, Twitter, Mail } from "lucide-react";
import { siteConfig, socialLinks } from "@/lib/constants";

const iconMap: Record<string, React.ReactNode> = {
  github: <Github size={18} />,
  linkedin: <Linkedin size={18} />,
  twitter: <Twitter size={18} />,
  mail: <Mail size={18} />,
};

function KaggleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.281.18.046.149.034.233-.07.352l-6.871 6.765 7.26 8.166c.096.116.104.211.001.318z" />
    </svg>
  );
}

export function Footer() {
  return (
    <footer
      className="border-t"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-surface)" }}
    >
      <div className="mx-auto max-w-6xl px-6 py-12">
        <div className="flex flex-col items-center gap-6 md:flex-row md:justify-between">
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            &copy; {new Date().getFullYear()} {siteConfig.name}. Built with
            curiosity.
          </p>

          <div className="flex items-center gap-2">
            {socialLinks.map((link) => (
              <Link
                key={link.name}
                href={link.url}
                target={link.url.startsWith("mailto") ? undefined : "_blank"}
                rel={link.url.startsWith("mailto") ? undefined : "noopener noreferrer"}
                className="flex h-10 w-10 items-center justify-center rounded-lg transition-colors hover:opacity-80"
                style={{ color: "var(--text-muted)" }}
                aria-label={link.name}
              >
                {link.icon === "kaggle" ? <KaggleIcon /> : iconMap[link.icon] || null}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
