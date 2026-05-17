import { anthropic, MODELS } from "../claude";

const SYSTEM = `You normalize fact-check submissions into a single verifiable assertion.

Rules:
- Strip rhetorical framing, opinion, and emotional language.
- Resolve pronouns and ambiguous references when context allows.
- Keep dates, names, and numbers exactly as given.
- If the input is a question, restate it as an assertion.
- If the input contains multiple independent claims, return the most central one.
- Output only the normalized assertion as plain text. No preamble.`;

export async function normalizeClaim(raw: string): Promise<string> {
  const res = await anthropic.messages.create({
    model: MODELS.cheap,
    max_tokens: 300,
    system: SYSTEM,
    messages: [{ role: "user", content: raw }],
  });
  const block = res.content.find((b) => b.type === "text");
  return block && block.type === "text" ? block.text.trim() : raw.trim();
}
