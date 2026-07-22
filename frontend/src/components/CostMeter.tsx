import { Box, Paper, Stack, Tooltip, Typography } from "@mui/material";
import type { AgentEvent } from "../lib/types";
import { AGENT, MONO } from "../theme";

const ORDER = ["router", "planner", "analyst", "critic", "composer", "system"];

/** Per-run cost telemetry: total, tokens, and a cost-by-agent stacked bar. */
export function CostMeter({ events }: { events: AgentEvent[] }) {
  const total = events.reduce((a, e) => a + e.cost_usd, 0);
  const tokens = events.reduce((a, e) => a + e.tokens_in + e.tokens_out, 0);

  const byAgent = new Map<string, number>();
  for (const e of events) {
    if (e.cost_usd > 0) byAgent.set(e.agent, (byAgent.get(e.agent) ?? 0) + e.cost_usd);
  }
  const segments = ORDER.filter((a) => byAgent.has(a)).map((a) => ({
    agent: a,
    cost: byAgent.get(a)!,
    color: AGENT[a] ?? AGENT.system,
  }));

  return (
    <Paper variant="outlined" sx={{ p: 1.75 }}>
      <Stack direction="row" alignItems="baseline" spacing={1} sx={{ mb: total > 0 ? 1.25 : 0 }}>
        <Typography sx={{ fontFamily: MONO, fontSize: "1.05rem", fontWeight: 600, color: "text.primary" }}>
          ${total.toFixed(4)}
        </Typography>
        <Typography sx={{ fontFamily: MONO, fontSize: "0.75rem", color: "text.disabled" }}>
          {tokens.toLocaleString()} tokens
        </Typography>
        <Box sx={{ flex: 1 }} />
        {total === 0 && events.length > 0 && (
          <Typography variant="caption" sx={{ color: "text.disabled" }}>
            free run · stub or replay
          </Typography>
        )}
      </Stack>

      {total > 0 && (
        <>
          <Box sx={{ display: "flex", height: 6, borderRadius: 999, overflow: "hidden", gap: "2px" }}>
            {segments.map((s) => (
              <Tooltip key={s.agent} title={`${s.agent} · $${s.cost.toFixed(4)}`}>
                <Box
                  sx={{
                    width: `${(s.cost / total) * 100}%`,
                    bgcolor: s.color,
                    transition: "width 400ms var(--ease-out)",
                    "&:first-of-type": { borderTopLeftRadius: 999, borderBottomLeftRadius: 999 },
                    "&:last-of-type": { borderTopRightRadius: 999, borderBottomRightRadius: 999 },
                  }}
                />
              </Tooltip>
            ))}
          </Box>
          <Stack direction="row" spacing={1.5} flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
            {segments.map((s) => (
              <Stack key={s.agent} direction="row" spacing={0.75} alignItems="center">
                <Box sx={{ width: 7, height: 7, borderRadius: "50%", bgcolor: s.color }} />
                <Typography sx={{ fontSize: "0.72rem", color: "text.secondary" }}>{s.agent}</Typography>
                <Typography sx={{ fontFamily: MONO, fontSize: "0.72rem", color: "text.disabled" }}>
                  ${s.cost.toFixed(4)}
                </Typography>
              </Stack>
            ))}
          </Stack>
        </>
      )}
    </Paper>
  );
}
