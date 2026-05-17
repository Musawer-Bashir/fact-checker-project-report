import { NextRequest } from "next/server";
import { runPipeline } from "@/lib/pipeline";
import {
  hashClaim,
  lookupCachedVerdict,
  persistRun,
} from "@/lib/db/supabase";
import type { Evidence, PipelineEvent, Verdict } from "@/lib/pipeline";

export const runtime = "nodejs";
export const maxDuration = 60;

function sse(event: object): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const claim = typeof body.claim === "string" ? body.claim.trim() : "";

  if (claim.length < 10) {
    return new Response(
      JSON.stringify({ error: "Claim must be at least 10 characters." }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }
  if (claim.length > 1000) {
    return new Response(
      JSON.stringify({ error: "Claim is too long (max 1000 characters)." }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (e: PipelineEvent | { type: "cached"; hash: string }) =>
        controller.enqueue(encoder.encode(sse(e)));

      let normalized = "";
      let evidence: Evidence[] = [];
      let verdict: Verdict | null = null;

      try {
        for await (const event of runPipeline(claim)) {
          send(event);
          if (event.type === "normalized") {
            normalized = event.text;
            const hash = hashClaim(normalized);
            const cached = await lookupCachedVerdict(hash);
            if (cached) {
              send({ type: "cached", hash });
              send({ type: "verdict", verdict: cached.verdict });
              send({ type: "stage", stage: "synthesize", status: "done" });
              controller.close();
              return;
            }
          }
          if (event.type === "evidence") evidence = event.items;
          if (event.type === "verdict") verdict = event.verdict;
        }

        if (normalized && verdict) {
          await persistRun({
            hash: hashClaim(normalized),
            raw: claim,
            normalized,
            evidence,
            verdict,
          });
        }
      } catch (err) {
        send({
          type: "error",
          message: err instanceof Error ? err.message : "Unknown error",
        });
      }
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
