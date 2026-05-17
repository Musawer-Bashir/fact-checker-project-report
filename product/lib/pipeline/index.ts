import { normalizeClaim } from "./normalize";
import { decomposeClaim } from "./decompose";
import { retrieveEvidence } from "./retrieve";
import { synthesizeVerdict } from "./synthesize";
import type { PipelineEvent } from "./types";

export async function* runPipeline(
  rawClaim: string,
): AsyncGenerator<PipelineEvent> {
  try {
    yield { type: "stage", stage: "normalize", status: "start" };
    const normalized = await normalizeClaim(rawClaim);
    yield { type: "normalized", text: normalized };
    yield { type: "stage", stage: "normalize", status: "done" };

    yield { type: "stage", stage: "decompose", status: "start" };
    const subclaims = await decomposeClaim(normalized);
    yield { type: "subclaims", items: subclaims };
    yield { type: "stage", stage: "decompose", status: "done" };

    yield { type: "stage", stage: "retrieve", status: "start" };
    const evidence = await retrieveEvidence(subclaims);
    yield { type: "evidence", items: evidence };
    yield { type: "stage", stage: "retrieve", status: "done" };

    yield { type: "stage", stage: "synthesize", status: "start" };
    const verdict = await synthesizeVerdict(normalized, subclaims, evidence);
    yield { type: "verdict", verdict };
    yield { type: "stage", stage: "synthesize", status: "done" };
  } catch (err) {
    yield {
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

export * from "./types";
