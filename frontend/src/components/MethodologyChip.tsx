import ScienceOutlinedIcon from "@mui/icons-material/ScienceOutlined";
import { Box, Tooltip } from "@mui/material";
import type { Methodology } from "../lib/types";
import { MONO } from "../theme";

const fmtP = (p: number) => (p < 0.001 ? "p<0.001" : `p=${p.toFixed(3)}`);

/** test used, n, p-value, effect size — the rigor receipt on a statistical claim. */
export function MethodologyChip({ m }: { m: Methodology }) {
  const parts = [m.method, `n=${m.n}`];
  if (m.p_value !== null) parts.push(fmtP(m.p_value));
  if (m.effect_size !== null)
    parts.push(`${m.effect_size_name || "effect"}=${m.effect_size.toFixed(2)}`);
  const chip = (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.75,
        px: 1,
        py: "3px",
        borderRadius: 1.5,
        border: "1px solid var(--border-strong)",
        bgcolor: "var(--surface-2)",
        color: "var(--ink-2)",
        fontFamily: MONO,
        fontSize: "0.68rem",
        whiteSpace: "nowrap",
        cursor: m.assumptions.length > 0 ? "help" : "default",
      }}
    >
      <ScienceOutlinedIcon sx={{ fontSize: 13, color: "var(--agent-critic)" }} />
      {parts.join(" · ")}
    </Box>
  );
  return m.assumptions.length > 0 ? (
    <Tooltip title={`Assumptions checked: ${m.assumptions.join("; ")}`}>{chip}</Tooltip>
  ) : (
    chip
  );
}
