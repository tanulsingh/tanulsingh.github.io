export const siteConfig = {
  name: "Tanul Singh",
  title: "Tanul Singh — ML Engineer & Researcher",
  description:
    "Senior ML Engineer at Apple, Kaggle Grandmaster, and self-taught AI practitioner. From a small town in India to building AI at scale — documenting the journey so others don't have to figure it out alone.",
  url: "https://tanulsingh.github.io",
  author: "Tanul Singh",
  handle: "Mr. KnowNothing",
  email: "tanulsingh0077@gmail.com",
};

export const navItems = [
  { label: "Blog", href: "/blog" },
  { label: "Papers", href: "/papers" },
  { label: "Projects", href: "/projects" },
  { label: "About", href: "/about" },
] as const;

export const socialLinks = [
  {
    name: "GitHub",
    url: "https://github.com/tanulsingh",
    icon: "github",
  },
  {
    name: "LinkedIn",
    url: "https://www.linkedin.com/in/tanul-singh/",
    icon: "linkedin",
  },
  {
    name: "Kaggle",
    url: "https://www.kaggle.com/tanulsingh077",
    icon: "kaggle",
  },
  {
    name: "Twitter / X",
    url: "https://x.com/singh_tanul",
    icon: "twitter",
  },
  {
    name: "Email",
    url: "mailto:tanulsingh0077@gmail.com",
    icon: "mail",
  },
] as const;

export const highlights = [
  {
    label: "Senior ML Engineer",
    detail: "Apple — multi-agentic systems & LLM research",
    icon: "cpu",
  },
  {
    label: "Kaggle Grandmaster",
    detail: "Notebooks GM, Competitions Master",
    icon: "trophy",
  },
  {
    label: "US Patent Holder",
    detail: "Dynamic intent detection system",
    icon: "scroll",
  },
  {
    label: "Self-Taught",
    detail: "ME degree → ML through sheer will",
    icon: "flame",
  },
] as const;
