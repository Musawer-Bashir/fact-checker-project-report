export default function About() {
  return (
    <article className="prose space-y-6">
      <h1 className="text-3xl tracking-tight">About GeoFact</h1>

      <section className="space-y-3 font-sans text-sm leading-relaxed">
        <p>
          GeoFact specializes in geopolitical claims — US-China relations, Taiwan,
          sanctions, trade, military activity, technology export controls, and
          adjacent topics. Geopolitics has its own source ecosystem, and we lean on
          it: think tanks like CSIS, RAND, Brookings, and Carnegie; outlets like
          Reuters, FT, Nikkei Asia, and SCMP; and primary documents from State, DoD,
          and equivalent bodies.
        </p>

        <h2 className="pt-4 text-lg text-[var(--foreground)]">How it works</h2>
        <ol className="list-decimal space-y-1 pl-5">
          <li>Your claim is normalized into a single verifiable assertion.</li>
          <li>Compound claims are split into atomic sub-claims.</li>
          <li>
            Each sub-claim is searched against our trusted-source allowlist (web
            search plus Wikipedia background).
          </li>
          <li>
            Claude Sonnet 4.6 reasons over the retrieved passages and produces a
            multi-class verdict with per-source citations.
          </li>
        </ol>

        <h2 className="pt-4 text-lg text-[var(--foreground)]">Limitations</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            Verdicts are AI-generated. Use them as a starting point and always read
            the cited sources.
          </li>
          <li>
            Outside our geopolitics focus, accuracy drops. Try Perplexity, Grok, or
            your favorite general fact-checker instead.
          </li>
          <li>
            We abstain ("Unverifiable" / "No Evidence") rather than guess. This is
            deliberate.
          </li>
        </ul>

        <h2 className="pt-4 text-lg text-[var(--foreground)]">Research lineage</h2>
        <p>
          The methodology behind GeoFact builds on a BSc Computer Science final-year
          project at the University of West London (2026), which used BERT and SHAP
          to study how token-level explanations affect user trust in fact-checking
          (n=15, Cohen's d=1.06, p=0.0024). The product version trades the classifier
          for an LLM-plus-retrieval architecture, but the design principle —
          <em> show sources prominently because explanations improve trust</em> —
          comes directly from that work.
        </p>
      </section>
    </article>
  );
}
