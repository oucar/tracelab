import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { Chip, Tooltip } from "@mui/material";
import type { VerifiedClaim } from "../lib/types";

/** verified (critic reconciled it) or unverified (with the discrepancy shown). */
export function ClaimBadge({ vc }: { vc: VerifiedClaim }) {
  const verified = vc.status === "verified";
  const chip = (
    <Chip
      size="small"
      variant="outlined"
      color={verified ? "success" : "warning"}
      icon={verified ? <CheckCircleOutlineIcon /> : <WarningAmberIcon />}
      label={vc.status}
    />
  );
  return verified ? chip : <Tooltip title={vc.detail || "unverified"}>{chip}</Tooltip>;
}
