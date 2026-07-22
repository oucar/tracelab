import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import WarningRoundedIcon from "@mui/icons-material/WarningRounded";
import { Box, Paper, Stack, Typography } from "@mui/material";
import type { FinalAnswer } from "../lib/types";
import { MONO } from "../theme";
import { ChartSpecRenderer } from "./ChartSpecRenderer";
import { ClaimBadge } from "./ClaimBadge";
import { MethodologyChip } from "./MethodologyChip";

/** The composed answer: narrative, per-claim verification ledger, and charts. */
export function AnswerPanel({ final }: { final: FinalAnswer }) {
  const verified = final.claims.filter((c) => c.status === "verified").length;
  const total = final.claims.length;
  const allVerified = total > 0 && verified === total;
  const accent = final.failed ? "var(--unverified)" : allVerified ? "var(--verified)" : "var(--unverified)";

  return (
    <Stack spacing={2}>
      {/* Narrative answer with a verification-status header. */}
      <Paper
        variant="outlined"
        sx={{ p: 0, overflow: "hidden", borderColor: `${accent}33` }}
      >
        <Stack
          direction="row"
          spacing={1}
          alignItems="center"
          sx={{ px: 2, py: 1, bgcolor: `${accent}0f`, borderBottom: `1px solid ${accent}22` }}
        >
          {final.failed ? (
            <WarningRoundedIcon sx={{ fontSize: 18, color: accent }} />
          ) : (
            <CheckCircleRoundedIcon sx={{ fontSize: 18, color: accent }} />
          )}
          <Typography variant="subtitle2" sx={{ color: accent, flex: 1 }}>
            {final.failed ? "answer · unresolved" : "answer"}
          </Typography>
          {total > 0 && (
            <Typography sx={{ fontFamily: MONO, fontSize: "0.7rem", color: accent }}>
              {verified}/{total} verified
            </Typography>
          )}
        </Stack>
        <Typography
          variant="body1"
          sx={{ px: 2, py: 1.75, color: "text.primary", maxWidth: "72ch" }}
        >
          {final.narrative}
        </Typography>
      </Paper>

      {/* Verification ledger. */}
      {total > 0 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1.25 }}>
            Claims · verified against independent re-derivation
          </Typography>
          <Stack divider={<Box sx={{ borderTop: "1px solid var(--border)" }} />}>
            {final.claims.map((vc) => (
              <Stack
                key={vc.claim.id}
                direction="row"
                spacing={1.5}
                alignItems="center"
                flexWrap="wrap"
                useFlexGap
                sx={{ py: 1 }}
              >
                <Typography variant="body2" sx={{ flex: "1 1 240px", color: "text.primary" }}>
                  {vc.claim.text}
                  {vc.claim.value !== null && (
                    <Box
                      component="span"
                      sx={{ fontFamily: MONO, color: "var(--accent-bright)", ml: 0.75, fontWeight: 500 }}
                    >
                      {String(vc.claim.value)}
                    </Box>
                  )}
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
