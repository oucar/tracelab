import ScienceOutlinedIcon from "@mui/icons-material/ScienceOutlined";
import { Chip, Tooltip } from "@mui/material";
import type { Methodology } from "../lib/types";

const fmtP = (p: number) => (p < 0.001 ? "p<0.001" : `p=${p.toFixed(3)}`);

/** test used, n, p-value, effect size — the rigor receipt on a statistical claim. */
export function MethodologyChip({ m }: { m: Methodology }) {
  const parts = [m.method, `n=${m.n}`];
  if (m.p_value !== null) parts.push(fmtP(m.p_value));
  if (m.effect_size !== null)
    parts.push(`${m.effect_size_name || "effect"}=${m.effect_size.toFixed(2)}`);
  const chip = (
    <Chip
      size="small"
      variant="outlined"
      icon={<ScienceOutlinedIcon />}
      label={parts.join(" · ")}
    />
  );
  return m.assumptions.length > 0 ? (
    <Tooltip title={`Assumptions checked: ${m.assumptions.join("; ")}`}>{chip}</Tooltip>
  ) : (
    chip
  );
}
