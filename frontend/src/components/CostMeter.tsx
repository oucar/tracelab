import { Chip, Paper, Stack, Typography } from "@mui/material";
import type { AgentEvent } from "../lib/types";

/** Per-run cost rollup with per-agent breakdown, derived live from events. */
export function CostMeter({ events }: { events: AgentEvent[] }) {
  const total = events.reduce((a, e) => a + e.cost_usd, 0);
  const tokens = events.reduce((a, e) => a + e.tokens_in + e.tokens_out, 0);
  const byAgent = new Map<string, number>();
  for (const e of events) {
    if (e.cost_usd > 0) byAgent.set(e.agent, (byAgent.get(e.agent) ?? 0) + e.cost_usd);
  }
  return (
    <Paper variant="outlined" sx={{ p: 1.5 }}>
      <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
        <Typography variant="subtitle2">
          ${total.toFixed(4)} · {tokens.toLocaleString()} tokens
        </Typography>
        {[...byAgent.entries()].map(([agent, cost]) => (
          <Chip key={agent} size="small" variant="outlined" label={`${agent} $${cost.toFixed(4)}`} />
        ))}
        {total === 0 && events.length > 0 && (
          <Typography variant="caption" color="text.secondary">
            free run (stub or replay)
          </Typography>
        )}
      </Stack>
    </Paper>
  );
}
