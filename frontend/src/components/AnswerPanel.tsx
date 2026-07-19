import { Alert, Paper, Stack, Typography } from "@mui/material";
import type { FinalAnswer } from "../lib/types";
import { ChartSpecRenderer } from "./ChartSpecRenderer";
import { ClaimBadge } from "./ClaimBadge";
import { MethodologyChip } from "./MethodologyChip";

/** The composed answer: narrative, per-claim verification badges, and charts. */
export function AnswerPanel({ final }: { final: FinalAnswer }) {
  return (
    <Stack spacing={2}>
      <Alert severity={final.failed ? "warning" : "success"}>{final.narrative}</Alert>

      {final.claims.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Claims
          </Typography>
          <Stack spacing={1}>
            {final.claims.map((vc) => (
              <Stack
                key={vc.claim.id}
                direction="row"
                spacing={1}
                alignItems="center"
                flexWrap="wrap"
                useFlexGap
              >
                <Typography variant="body2" sx={{ flex: "1 1 240px" }}>
                  {vc.claim.text}
                  {vc.claim.value !== null && ` — ${vc.claim.value}`}
                </Typography>
                {vc.claim.methodology && <MethodologyChip m={vc.claim.methodology} />}
                <ClaimBadge vc={vc} />
              </Stack>
            ))}
          </Stack>
        </Paper>
      )}

      {final.charts.map((spec, i) => (
        <ChartSpecRenderer key={i} spec={spec} />
      ))}
    </Stack>
  );
}
