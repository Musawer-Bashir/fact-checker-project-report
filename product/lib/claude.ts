import Anthropic from "@anthropic-ai/sdk";

export const anthropic = new Anthropic();

export const MODELS = {
  cheap: "claude-haiku-4-5",
  smart: "claude-sonnet-4-6",
} as const;
