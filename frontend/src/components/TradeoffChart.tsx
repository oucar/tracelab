import { Paper, Typography } from "@mui/material";
import { ScatterChart } from "@mui/x-charts/ScatterChart";
import type { EvalRunSummary } from "../lib/types";

/**
 * Cost/quality tradeoff scatter for `study:*` eval runs — one point per config,
 * keeping only the newest run per config label. Renders nothing when no study
 * runs exist (the current state while the live study is skipped).
 */
export function TradeoffChart({ runs }: { runs: EvalRunSummary[] }) {
  const latest = new Map<string, EvalRunSummary>();
  for (const run of [...runs].sort((a, b) => b.created_at - a.created_at)) {
    if (!run.label.startsWith("study:")) continue;
    const config = run.label.slice("study:".length);
    if (!latest.has(config)) latest.set(config, run);
  }
  if (latest.size === 0) return null;

  const series = [...latest.entries()].map(([config, run]) => {
    const perQ = run.cost_usd / Math.max(run.questions_total, 1);
    const quality = run.judge_avg ?? (run.tier1_pass_rate ?? 0) * 5;
    return { label: config, data: [{ id: run.id, x: perQ, y: quality }] };
  });

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Quality vs cost (latest study run per config)
      </Typography>
      <ScatterChart
        height={300}
        series={series}
        xAxis={[{ label: "cost per question (USD)" }]}
        yAxis={[{ label: "quality (judge avg 1-5)", min: 1, max: 5 }]}
      />
    </Paper>
  );
}
