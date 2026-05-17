import { TRUSTED_DOMAINS, domainOf, isTrustedDomain } from "../sources/geopolitics-allowlist";
import type { Evidence } from "./types";

interface TavilyResult {
  url: string;
  title: string;
  content: string;
  score: number;
}

interface TavilyResponse {
  results: TavilyResult[];
}

async function tavilySearch(query: string, maxResults = 8): Promise<TavilyResult[]> {
  const key = process.env.TAVILY_API_KEY;
  if (!key) throw new Error("TAVILY_API_KEY not set");

  const res = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: key,
      query,
      search_depth: "advanced",
      max_results: maxResults,
      include_domains: TRUSTED_DOMAINS,
    }),
  });

  if (!res.ok) {
    throw new Error(`Tavily error ${res.status}: ${await res.text()}`);
  }
  const json = (await res.json()) as TavilyResponse;
  return json.results ?? [];
}

async function wikipediaSearch(query: string, maxResults = 3): Promise<TavilyResult[]> {
  const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&srlimit=${maxResults}&format=json&origin=*`;
  const res = await fetch(url);
  if (!res.ok) return [];
  const json = (await res.json()) as {
    query?: { search?: Array<{ title: string; snippet: string }> };
  };
  const hits = json.query?.search ?? [];
  return hits.map((h) => ({
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(h.title.replace(/ /g, "_"))}`,
    title: h.title,
    content: h.snippet.replace(/<[^>]+>/g, ""),
    score: 0.5,
  }));
}

export async function retrieveEvidence(subclaims: string[]): Promise<Evidence[]> {
  const seen = new Set<string>();
  const out: Evidence[] = [];

  const results = await Promise.all(
    subclaims.map(async (claim) => {
      const [tavily, wiki] = await Promise.allSettled([
        tavilySearch(claim, 6),
        wikipediaSearch(claim, 2),
      ]);
      const merged: TavilyResult[] = [];
      if (tavily.status === "fulfilled") merged.push(...tavily.value);
      if (wiki.status === "fulfilled") merged.push(...wiki.value);
      return merged;
    }),
  );

  for (const list of results) {
    for (const r of list) {
      if (seen.has(r.url)) continue;
      seen.add(r.url);
      out.push({
        url: r.url,
        title: r.title,
        domain: domainOf(r.url),
        snippet: r.content,
        trusted: isTrustedDomain(r.url),
      });
    }
  }

  out.sort((a, b) => Number(b.trusted) - Number(a.trusted));
  return out.slice(0, 12);
}
