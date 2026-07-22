import { Box, Tooltip } from "@mui/material";
import type { VerifiedClaim } from "../lib/types";

/** verified (critic reconciled it) or unverified (with the discrepancy shown). */
export function ClaimBadge({ vc }: { vc: VerifiedClaim }) {
  const verified = vc.status === "verified";
  const color = verified ? "var(--verified)" : "var(--unverified)";
  const badge = (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.75,
        px: 1,
        py: "3px",
        borderRadius: 1.5,
        border: `1px solid ${color}44`,
        bgcolor: `${color}14`,
        color,
        fontSize: "0.7rem",
        fontWeight: 600,
        letterSpacing: "0.01em",
        whiteSpace: "nowrap",
        cursor: verified ? "default" : "help",
      }}
    >
      <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: color }} />
      {vc.status}
    </Box>
  );
  return verified ? badge : <Tooltip title={vc.detail || "unverified"}>{badge}</Tooltip>;
}
