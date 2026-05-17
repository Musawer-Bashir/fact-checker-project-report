# GeoFact (product)

A specialized fact-checker for US-China geopolitical claims. Decomposes the claim, retrieves evidence from a curated allowlist of trusted sources, and uses Claude Sonnet 4.6 to synthesize a multi-class verdict with per-source citations.

This lives alongside the BSc research artifact at the repo root, which is preserved unchanged.

## Stack

- Next.js 15 (App Router) + React 19 + Tailwind v4
- `@anthropic-ai/sdk` — Claude Sonnet 4.6 (synthesis) + Haiku 4.5 (normalize, decompose)
- Tavily API (web search, restricted to a trusted-source allowlist) + Wikipedia API (background)
- Supabase Postgres (claim logging + 24h verdict cache)
- Deploys to Vercel

## Setup

```bash
cd product
pnpm install   # or npm/yarn

cp .env.example .env.local
# Fill in:
#   ANTHROPIC_API_KEY        from console.anthropic.com
#   TAVILY_API_KEY           from tavily.com (free tier covers MVP)
#   NEXT_PUBLIC_SUPABASE_URL + keys from supabase.com project settings
```

Apply the schema via the Supabase SQL editor or CLI:

```bash
supabase db push   # or paste supabase/migrations/0001_init.sql into the SQL editor
```

## Run

```bash
pnpm dev          # http://localhost:3000
pnpm typecheck    # type errors only
pnpm eval         # run the 30-claim eval set against the live pipeline
pnpm eval tw-un-1 # run a single eval case
```

## Pipeline

```
raw claim
  → normalize    (Haiku 4.5)  — strip rhetoric, single assertion
  → decompose    (Haiku 4.5)  — split into 1-4 atomic sub-claims
  → retrieve     (Tavily + Wikipedia, domain-allowlisted)
  → synthesize   (Sonnet 4.6, structured output)
  → 7-class verdict + per-source citations, streamed via SSE
```

Verdict labels: `True` · `Mostly True` · `Mixed` · `Mostly False` · `False` · `Unverifiable` · `No Evidence`.

The synthesis prompt explicitly prefers abstention ("Unverifiable" / "No Evidence") over guessing — calibration matters more than coverage for a fact-checker.

## File map

```
app/
  page.tsx              landing + input
  api/check/route.ts    streaming endpoint (SSE)
  c/[hash]/page.tsx     shareable verdict permalink
  about/page.tsx        methodology + research lineage
lib/
  claude.ts             Anthropic client + model IDs
  pipeline/
    normalize.ts        Haiku 4.5
    decompose.ts        Haiku 4.5 + json_schema output
    retrieve.ts         Tavily + Wikipedia
    synthesize.ts       Sonnet 4.6 + json_schema output, with prompt caching
    types.ts            shared interfaces
    index.ts            async generator yielding PipelineEvents
  sources/
    geopolitics-allowlist.ts   ~40 trusted domains
  db/supabase.ts        claim logging + cache lookup
components/
  ClaimChecker.tsx      single client component, streams from /api/check
supabase/migrations/
  0001_init.sql         claims + verdicts tables
tests/
  eval-set.json         30 hand-curated US-China claims with ground truth
  eval.ts               runs the pipeline against eval-set, reports accuracy
```

## Deploy

```bash
vercel
# Add ANTHROPIC_API_KEY, TAVILY_API_KEY, NEXT_PUBLIC_SUPABASE_URL,
# NEXT_PUBLIC_SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY in the Vercel dashboard.
```

The `/api/check` route streams Server-Sent Events. `maxDuration` is set to 60s, so plan in 6-10s per stage. For longer-running checks, switch the runtime to a long-running deployment.

## What's deliberately not here yet

Auth, accounts, team features, browser extension, public API, mobile app, multilingual support, image/URL submission. Add when usage data justifies them.
