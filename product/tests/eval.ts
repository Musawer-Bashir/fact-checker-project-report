import { runPipeline } from "../lib/pipeline";
import evalSet from "./eval-set.json";

interface EvalCase {
  id: string;
  text: string;
  expected: string;
  category: string;
}

interface Result {
  id: string;
  category: string;
  expected: string;
  actual: string;
  confidence: number;
  match: boolean;
  durationMs: number;
}

function near(expected: string, actual: string): boolean {
  if (expected === actual) return true;
  const truth = new Set(["True", "Mostly True"]);
  const falsy = new Set(["False", "Mostly False"]);
  if (truth.has(expected) && truth.has(actual)) return true;
  if (falsy.has(expected) && falsy.has(actual)) return true;
  return false;
}

async function runOne(c: EvalCase): Promise<Result> {
  const t0 = Date.now();
  let actual = "ERROR";
  let confidence = 0;
  for await (const event of runPipeline(c.text)) {
    if (event.type === "verdict") {
      actual = event.verdict.label;
      confidence = event.verdict.confidence;
    }
    if (event.type === "error") {
      actual = `ERROR: ${event.message}`;
    }
  }
  return {
    id: c.id,
    category: c.category,
    expected: c.expected,
    actual,
    confidence,
    match: near(c.expected, actual),
    durationMs: Date.now() - t0,
  };
}

async function main() {
  const cases = (evalSet as { claims: EvalCase[] }).claims;
  const onlyId = process.argv[2];
  const filtered = onlyId ? cases.filter((c) => c.id === onlyId) : cases;

  console.log(`Running ${filtered.length} eval cases…\n`);
  const results: Result[] = [];
  for (const c of filtered) {
    const r = await runOne(c);
    results.push(r);
    const mark = r.match ? "✓" : "✗";
    console.log(
      `${mark} ${r.id.padEnd(12)} expected=${r.expected.padEnd(13)} actual=${r.actual.padEnd(13)} conf=${(r.confidence * 100).toFixed(0)}% (${r.durationMs}ms)`,
    );
  }

  const matches = results.filter((r) => r.match).length;
  const highConfWrong = results.filter((r) => !r.match && r.confidence > 0.8);
  console.log(
    `\n${matches}/${results.length} matched (${((matches / results.length) * 100).toFixed(0)}%)`,
  );
  if (highConfWrong.length > 0) {
    console.log(`\n⚠️  ${highConfWrong.length} wrong with confidence > 80%:`);
    for (const r of highConfWrong) {
      console.log(`   ${r.id}: ${r.expected} → ${r.actual} (${(r.confidence * 100).toFixed(0)}%)`);
    }
  }
  process.exit(highConfWrong.length > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
