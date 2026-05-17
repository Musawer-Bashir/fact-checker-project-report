import { anthropic, MODELS } from "../claude";

const SYSTEM = `You split a normalized claim into 1-4 atomic sub-claims that can each be independently verified.

Rules:
- An atomic sub-claim contains a single factual assertion.
- If the claim is already atomic, return just the original claim.
- Prefer fewer sub-claims. Never produce more than 4.
- Keep names, dates, and numbers intact.
- Do not introduce new assertions not implied by the original.`;

const SCHEMA = {
  type: "object",
  properties: {
    subclaims: {
      type: "array",
      items: { type: "string" },
      minItems: 1,
      maxItems: 4,
    },
  },
  required: ["subclaims"],
  additionalProperties: false,
} as const;

export async function decomposeClaim(normalized: string): Promise<string[]> {
  const res = await anthropic.messages.create({
    model: MODELS.cheap,
    max_tokens: 500,
    system: SYSTEM,
    output_config: {
      format: { type: "json_schema", schema: SCHEMA },
    },
    messages: [{ role: "user", content: normalized }],
  });
  const block = res.content.find((b) => b.type === "text");
  if (!block || block.type !== "text") return [normalized];
  try {
    const parsed = JSON.parse(block.text) as { subclaims: string[] };
    return parsed.subclaims.length > 0 ? parsed.subclaims : [normalized];
  } catch {
    return [normalized];
  }
}
