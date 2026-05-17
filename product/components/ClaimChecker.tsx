"use client";

import { useState, type FormEvent } from "react";
import type { Evidence, PipelineEvent, Verdict } from "@/lib/pipeline";

type Stage = "normalize" | "decompose" | "retrieve" | "synthesize";
type StageState = "pending" | "running" | "done";

const STAGE_LABELS: Record<Stage, string> = {
  normalize: "Normalizing the claim",
  decompose: "Breaking into sub-claims",
  retrieve: "Searching sources",
  synthesize: "Reasoning over evidence",
};

const VERDICT_COLOR: Record<Verdict["label"], string> = {
  True: "bg-emerald-100 text-emerald-900 border-emerald-300",
  "Mostly True": "bg-emerald-50 text-emerald-900 border-emerald-200",
  Mixed: "bg-amber-100 text-amber-900 border-amber-300",
  "Mostly False": "bg-orange-50 text-orange-900 border-orange-200",
  False: "bg-rose-100 text-rose-900 border-rose-300",
  Unverifiable: "bg-slate-100 text-slate-900 border-slate-300",
  "No Evidence": "bg-slate-50 text-slate-700 border-slate-200",
};

interface State {
  busy: boolean;
  stages: Record<Stage, StageState>;
  normalized?: string;
  subclaims?: string[];
  evidence?: Evidence[];
  verdict?: Verdict;
  cached?: boolean;
  error?: string;
}

const initialState: State = {
  busy: false,
  stages: {
    normalize: "pending",
    decompose: "pending",
    retrieve: "pending",
    synthesize: "pending",
  },
};

export function ClaimChecker() {
  const [claim, setClaim] = useState("");
  const [state, setState] = useState<State>(initialState);

  async function check(e: FormEvent) {
    e.preventDefault();
    if (claim.trim().length < 10) return;
    setState({ ...initialState, busy: true });

    const res = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim }),
    });

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "Request failed");
      setState((s) => ({ ...s, busy: false, error: text }));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        try {
          const event = JSON.parse(line.slice(6)) as
            | PipelineEvent
            | { type: "cached"; hash: string };
          applyEvent(setState, event);
        } catch {
          // ignore parse errors on partial chunks
        }
      }
    }
    setState((s) => ({ ...s, busy: false }));
  }

  return (
    <div className="space-y-8">
      <form onSubmit={check} className="space-y-3">
        <label htmlFor="claim" className="block font-sans text-sm text-[var(--muted)]">
          Paste a claim — about US-China relations, Taiwan, trade, tech, military, etc.
        </label>
        <textarea
          id="claim"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          placeholder="e.g. The US Navy conducted a freedom of navigation operation near the Spratly Islands in 2024."
          rows={3}
          maxLength={1000}
          disabled={state.busy}
          className="w-full resize-none rounded-md border border-[var(--border)] bg-transparent p-3 font-sans text-base outline-none focus:border-[var(--accent)]"
        />
        <div className="flex items-center justify-between">
          <span className="font-sans text-xs text-[var(--muted)]">
            {claim.length}/1000
          </span>
          <button
            type="submit"
            disabled={state.busy || claim.trim().length < 10}
            className="rounded-md bg-[var(--foreground)] px-5 py-2 font-sans text-sm text-[var(--background)] disabled:opacity-40"
          >
            {state.busy ? "Checking…" : "Check claim"}
          </button>
        </div>
      </form>

      {(state.busy || state.verdict || state.error) && (
        <PipelineStream state={state} />
      )}

      {state.verdict && <VerdictCard verdict={state.verdict} cached={!!state.cached} />}

      {state.evidence && state.evidence.length > 0 && state.verdict && (
        <CitationList citations={state.verdict.citations} />
      )}

      {state.error && (
        <div className="rounded-md border border-rose-300 bg-rose-50 p-4 font-sans text-sm text-rose-900">
          {state.error}
        </div>
      )}
    </div>
  );
}

function applyEvent(
  set: (fn: (s: State) => State) => void,
  event: PipelineEvent | { type: "cached"; hash: string },
) {
  set((s) => {
    const next = { ...s, stages: { ...s.stages } };
    switch (event.type) {
      case "stage":
        next.stages[event.stage] = event.status === "start" ? "running" : "done";
        return next;
      case "normalized":
        next.normalized = event.text;
        return next;
      case "subclaims":
        next.subclaims = event.items;
        return next;
      case "evidence":
        next.evidence = event.items;
        return next;
      case "verdict":
        next.verdict = event.verdict;
        return next;
      case "cached":
        next.cached = true;
        return next;
      case "error":
        next.error = event.message;
        return next;
    }
  });
}

function PipelineStream({ state }: { state: State }) {
  const stages: Stage[] = ["normalize", "decompose", "retrieve", "synthesize"];
  return (
    <div className="rounded-md border border-[var(--border)] p-4 font-sans">
      <ol className="space-y-2 text-sm">
        {stages.map((stage) => {
          const st = state.stages[stage];
          return (
            <li key={stage} className="flex items-start gap-3">
              <span className="mt-0.5 inline-block h-4 w-4 flex-shrink-0">
                {st === "done" ? "✓" : st === "running" ? "•" : "○"}
              </span>
              <div className="flex-1">
                <div className={st === "pending" ? "text-[var(--muted)]" : ""}>
                  {STAGE_LABELS[stage]}
                </div>
                {stage === "normalize" && state.normalized && (
                  <div className="mt-1 italic text-[var(--muted)]">
                    "{state.normalized}"
                  </div>
                )}
                {stage === "decompose" && state.subclaims && state.subclaims.length > 1 && (
                  <ul className="mt-1 list-disc pl-4 text-[var(--muted)]">
                    {state.subclaims.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                )}
                {stage === "retrieve" && state.evidence && (
                  <div className="mt-1 text-xs text-[var(--muted)]">
                    {state.evidence.length} passages from{" "}
                    {new Set(state.evidence.map((e) => e.domain)).size} sources
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function VerdictCard({ verdict, cached }: { verdict: Verdict; cached: boolean }) {
  return (
    <div className={`rounded-md border-2 p-6 ${VERDICT_COLOR[verdict.label]}`}>
      <div className="mb-3 flex items-baseline justify-between font-sans">
        <span className="text-2xl font-semibold">{verdict.label}</span>
        <span className="text-sm">
          Confidence: {Math.round(verdict.confidence * 100)}%
          {cached && " · cached"}
        </span>
      </div>
      <p className="text-lg">{verdict.summary}</p>
      <p className="mt-3 text-sm opacity-80">{verdict.reasoning}</p>
    </div>
  );
}

function CitationList({ citations }: { citations: Verdict["citations"] }) {
  const grouped: Record<string, Verdict["citations"]> = {
    supports: [],
    refutes: [],
    context: [],
  };
  for (const c of citations) grouped[c.stance].push(c);

  return (
    <div className="space-y-4 font-sans">
      <h2 className="text-lg">Sources</h2>
      {(["supports", "refutes", "context"] as const).map((stance) =>
        grouped[stance].length === 0 ? null : (
          <div key={stance}>
            <h3 className="mb-2 text-xs uppercase tracking-wide text-[var(--muted)]">
              {stance === "supports"
                ? "Supports the claim"
                : stance === "refutes"
                  ? "Refutes the claim"
                  : "Context"}
            </h3>
            <ul className="space-y-2">
              {grouped[stance].map((c, i) => (
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
                  <div className="text-xs text-[var(--muted)]">{c.domain}</div>
                  <blockquote className="mt-2 border-l-2 border-[var(--border)] pl-3 italic">
                    "{c.quote}"
                  </blockquote>
                </li>
              ))}
            </ul>
          </div>
        ),
      )}
    </div>
  );
}
