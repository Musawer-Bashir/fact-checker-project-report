export const TRUSTED_DOMAINS = [
  "reuters.com",
  "apnews.com",
  "bbc.com",
  "bbc.co.uk",
  "ft.com",
  "economist.com",
  "wsj.com",
  "nytimes.com",
  "washingtonpost.com",
  "bloomberg.com",
  "nikkei.com",
  "asia.nikkei.com",
  "scmp.com",
  "japantimes.co.jp",
  "straitstimes.com",

  "csis.org",
  "rand.org",
  "brookings.edu",
  "cfr.org",
  "carnegieendowment.org",
  "hudson.org",
  "atlanticcouncil.org",
  "iiss.org",
  "merics.org",
  "stimson.org",
  "lowyinstitute.org",
  "chathamhouse.org",

  "state.gov",
  "defense.gov",
  "whitehouse.gov",
  "treasury.gov",
  "commerce.gov",
  "uscc.gov",
  "ustr.gov",
  "europa.eu",
  "gov.uk",
  "un.org",
  "imf.org",
  "worldbank.org",
  "oecd.org",
  "wto.org",
  "iaea.org",

  "en.wikipedia.org",
];

export function isTrustedDomain(url: string): boolean {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    return TRUSTED_DOMAINS.some((d) => host === d || host.endsWith(`.${d}`));
  } catch {
    return false;
  }
}

export function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}
