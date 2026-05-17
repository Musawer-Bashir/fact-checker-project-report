import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "GeoFact — fact-checking for US-China geopolitics",
  description:
    "Verify geopolitical claims about US-China relations, Taiwan, trade, and tech with cited evidence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="mx-auto max-w-3xl px-6 py-8">
          <header className="mb-12 flex items-baseline justify-between">
            <Link href="/" className="text-2xl tracking-tight">
              <span className="italic">Geo</span>Fact
            </Link>
            <nav className="font-sans text-sm text-[var(--muted)]">
              <Link href="/about" className="hover:text-[var(--foreground)]">
                About
              </Link>
            </nav>
          </header>
          <main>{children}</main>
          <footer className="mt-24 border-t border-[var(--border)] pt-6 font-sans text-xs text-[var(--muted)]">
            Built on top of a BSc research project at the University of West London.
            Verdicts are AI-generated from cited public sources — always check the
            originals.
          </footer>
        </div>
      </body>
    </html>
  );
}
