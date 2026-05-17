export type VerdictLabel =
  | "True"
  | "Mostly True"
  | "Mixed"
  | "Mostly False"
  | "False"
  | "Unverifiable"
  | "No Evidence";

export interface Evidence {
  url: string;
  title: string;
  domain: string;
  snippet: string;
  trusted: boolean;
}

export interface RankedEvidence extends Evidence {
  stance: "supports" | "refutes" | "context";
}

export interface Verdict {
  label: VerdictLabel;
  confidence: number;
  summary: string;
  reasoning: string;
  citations: Array<{
    url: string;
    title: string;
    domain: string;
    stance: "supports" | "refutes" | "context";
    quote: string;
  }>;
}

export type PipelineEvent =
  | { type: "stage"; stage: "normalize" | "decompose" | "retrieve" | "synthesize"; status: "start" | "done" }
  | { type: "normalized"; text: string }
  | { type: "subclaims"; items: string[] }
  | { type: "evidence"; items: Evidence[] }
  | { type: "verdict"; verdict: Verdict }
  | { type: "error"; message: string };
