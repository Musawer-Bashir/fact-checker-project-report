import { lookupCachedVerdict } from "@/lib/db/supabase";
import { notFound } from "next/navigation";

interface PageProps {
  params: Promise<{ hash: string }>;
}

const VERDICT_COLOR: Record<string, string> = {
  True: "bg-emerald-100 text-emerald-900 border-emerald-300",
  "Mostly True": "bg-emerald-50 text-emerald-900 border-emerald-200",
  Mixed: "bg-amber-100 text-amber-900 border-amber-300",
  "Mostly False": "bg-orange-50 text-orange-900 border-orange-200",
  False: "bg-rose-100 text-rose-900 border-rose-300",
  Unverifiable: "bg-slate-100 text-slate-900 border-slate-300",
  "No Evidence": "bg-slate-50 text-slate-700 border-slate-200",
};

export default async function VerdictPage({ params }: PageProps) {
  const { hash } = await params;
  const cached = await lookupCachedVerdict(hash);
  if (!cached) notFound();

  const { verdict, normalized, raw } = cached;

  return (
    <article className="space-y-8">
      <div>
        <div className="font-sans text-xs uppercase tracking-wide text-[var(--muted)]">
          Claim
        </div>
        <h1 className="mt-1 text-xl">{raw}</h1>
        {normalized !== raw && (
          <p className="mt-2 font-sans text-sm italic text-[var(--muted)]">
            Normalized as: "{normalized}"
          </p>
        )}
      </div>

      <div className={`rounded-md border-2 p-6 ${VERDICT_COLOR[verdict.label]}`}>
        <div className="mb-3 flex items-baseline justify-between font-sans">
          <span className="text-2xl font-semibold">{verdict.label}</span>
          <span className="text-sm">
            Confidence: {Math.round(verdict.confidence * 100)}%
          </span>
        </div>
        <p className="text-lg">{verdict.summary}</p>
        <p className="mt-3 text-sm opacity-80">{verdict.reasoning}</p>
      </div>

      <div className="space-y-2 font-sans">
        <h2 className="text-lg">Sources</h2>
        <ul className="space-y-2">
          {verdict.citations.map((c, i) => (
            <li
              key={i}
              className="rounded border border-[var(--border)] p-3 text-sm"
            >
              <a
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium hover:underline"
              >
                {c.title}
              </a>
              <div className="text-xs text-[var(--muted)]">
                {c.domain} · {c.stance}
              </div>
              <blockquote className="mt-2 border-l-2 border-[var(--border)] pl-3 italic">
                "{c.quote}"
              </blockquote>
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
}
