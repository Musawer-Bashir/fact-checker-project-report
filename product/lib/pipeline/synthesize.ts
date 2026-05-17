import { anthropic, MODELS } from "../claude";
import type { Evidence, Verdict } from "./types";

const SYSTEM = `You are a careful, neutral fact-checker for geopolitical claims, with particular expertise in US-China relations, Taiwan, trade policy, tech sanctions, and military affairs.

Your job: given a normalized claim and a set of evidence passages from cited sources, produce a verdict.

Rules:
- Ground every assertion in a specific cited source. If a source doesn't say it, don't claim it.
- Quote at most ~20 words verbatim from any single source.
- If sources disagree, return "Mixed" and explain the disagreement.
- If you cannot find sufficient evidence either way, return "Unverifiable" or "No Evidence" — do NOT guess.
- "No Evidence" = nothing in the provided sources addresses the claim. "Unverifiable" = sources address it but the answer is genuinely contested or unknowable from public information.
- Confidence is your calibrated probability that the verdict is correct (0.0 to 1.0). For "Unverifiable" or "No Evidence", confidence should be low (≤ 0.5).
- Be especially careful with claims involving classified intelligence, attributed quotes, and statistics — these are commonly misrepresented.`;

const SCHEMA = {
  type: "object",
  properties: {
    label: {
      type: "string",
      enum: [
        "True",
        "Mostly True",
        "Mixed",
        "Mostly False",
        "False",
        "Unverifiable",
        "No Evidence",
      ],
    },
    confidence: { type: "number" },
    summary: {
      type: "string",
      description: "1-2 sentence verdict summary the user reads first.",
    },
    reasoning: {
      type: "string",
      description: "2-4 sentences explaining how the evidence supports the verdict.",
    },
    citations: {
      type: "array",
      items: {
        type: "object",
        properties: {
          url: { type: "string" },
          title: { type: "string" },
          domain: { type: "string" },
          stance: {
            type: "string",
            enum: ["supports", "refutes", "context"],
          },
          quote: {
            type: "string",
            description: "Short verbatim quote from the source (≤ 20 words).",
          },
        },
        required: ["url", "title", "domain", "stance", "quote"],
        additionalProperties: false,
      },
      minItems: 1,
      maxItems: 8,
    },
  },
  required: ["label", "confidence", "summary", "reasoning", "citations"],
  additionalProperties: false,
} as const;

function formatEvidence(evidence: Evidence[]): string {
  return evidence
    .map(
      (e, i) =>
        `[${i + 1}] ${e.domain} — ${e.title}\n${e.snippet}\nURL: ${e.url}\n`,
    )
    .join("\n");
}

export async function synthesizeVerdict(
  normalizedClaim: string,
  subclaims: string[],
  evidence: Evidence[],
): Promise<Verdict> {
  if (evidence.length === 0) {
    return {
      label: "No Evidence",
      confidence: 0.2,
      summary: "No sources could be retrieved for this claim.",
      reasoning:
        "Search across our curated set of trusted geopolitical sources returned no matching passages. This may mean the claim is too vague, too new, or outside our domain.",
      citations: [],
    };
  }

  const userMessage = `Normalized claim:
${normalizedClaim}

Sub-claims to evaluate:
${subclaims.map((s, i) => `${i + 1}. ${s}`).join("\n")}

Evidence passages:
${formatEvidence(evidence)}

Now produce the verdict.`;

  const res = await anthropic.messages.create({
    model: MODELS.smart,
    max_tokens: 2000,
    system: [
      {
        type: "text",
        text: SYSTEM,
        cache_control: { type: "ephemeral" },
      },
    ],
    output_config: {
      format: { type: "json_schema", schema: SCHEMA },
    },
    messages: [{ role: "user", content: userMessage }],
  });

  const block = res.content.find((b) => b.type === "text");
  if (!block || block.type !== "text") {
    throw new Error("synthesize: no text block in response");
  }
  return JSON.parse(block.text) as Verdict;
}
