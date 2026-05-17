import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { createHash } from "node:crypto";
import type { Evidence, Verdict } from "../pipeline/types";

let adminClient: SupabaseClient | null = null;

function admin(): SupabaseClient | null {
  if (adminClient) return adminClient;
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;
  adminClient = createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  return adminClient;
}

export function hashClaim(normalized: string): string {
  return createHash("sha256")
    .update(normalized.toLowerCase().trim())
    .digest("hex")
    .slice(0, 16);
}

export async function lookupCachedVerdict(hash: string): Promise<{
  verdict: Verdict;
  normalized: string;
  raw: string;
} | null> {
  const sb = admin();
  if (!sb) return null;

  const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const { data, error } = await sb
    .from("claims")
    .select("raw_text, normalized_text, verdicts(label, confidence, summary, reasoning, citations)")
    .eq("hash", hash)
    .gte("created_at", since)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error || !data) return null;
  const verdictRow = Array.isArray(data.verdicts) ? data.verdicts[0] : data.verdicts;
  if (!verdictRow) return null;

  return {
    raw: data.raw_text,
    normalized: data.normalized_text,
    verdict: {
      label: verdictRow.label,
      confidence: verdictRow.confidence,
      summary: verdictRow.summary,
      reasoning: verdictRow.reasoning,
      citations: verdictRow.citations,
    },
  };
}

export async function persistRun(args: {
  hash: string;
  raw: string;
  normalized: string;
  evidence: Evidence[];
  verdict: Verdict;
}): Promise<void> {
  const sb = admin();
  if (!sb) return;

  const { data: claim, error: claimErr } = await sb
    .from("claims")
    .insert({
      raw_text: args.raw,
      normalized_text: args.normalized,
      hash: args.hash,
    })
    .select("id")
    .single();
  if (claimErr || !claim) return;

  await sb.from("verdicts").insert({
    claim_id: claim.id,
    label: args.verdict.label,
    confidence: args.verdict.confidence,
    summary: args.verdict.summary,
    reasoning: args.verdict.reasoning,
    citations: args.verdict.citations,
    evidence_count: args.evidence.length,
    model_version: "sonnet-4-6-v1",
  });
}
