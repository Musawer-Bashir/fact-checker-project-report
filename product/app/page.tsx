import { ClaimChecker } from "@/components/ClaimChecker";

export default function Home() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl tracking-tight">Fact-check a geopolitical claim</h1>
        <p className="mt-2 font-sans text-sm text-[var(--muted)]">
          GeoFact searches a curated set of news outlets, think tanks, and government
          sources, then reasons over what it finds. Every verdict cites its sources —
          always check them.
        </p>
      </div>
      <ClaimChecker />
    </div>
  );
}
