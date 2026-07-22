import {
  Box,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from "@mui/material";
import type { CalibrationDimension, CalibrationReport } from "../lib/types";

function Matrix({ dim }: { dim: CalibrationDimension }) {
  const max = Math.max(1, ...dim.matrix.flat());
  return (
    <Box>
      <Typography variant="caption" color="text.secondary">
        {dim.dimension} — human (rows) vs judge (cols), 1→5
      </Typography>
      <Box sx={{ display: "grid", gridTemplateColumns: "repeat(5, 22px)", gap: "2px", mt: 0.5 }}>
        {dim.matrix.flatMap((row, h) =>
          row.map((count, j) => (
            <Tooltip key={`${h}-${j}`} title={`human ${h + 1} / judge ${j + 1}: ${count}`}>
              <Box
                sx={{
                  width: 22,
                  height: 22,
                  borderRadius: 0.5,
                  bgcolor: count === 0 ? "action.hover" : "primary.main",
                  opacity: count === 0 ? 1 : 0.25 + 0.75 * (count / max),
                }}
              />
            </Tooltip>
          )),
        )}
      </Box>
    </Box>
  );
}

export function CalibrationGrid({
  report,
  loading,
}: {
  report: CalibrationReport | undefined;
  loading: boolean;
}) {
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Judge calibration
      </Typography>
      {loading && <Typography color="text.secondary">loading…</Typography>}
      {!loading && (!report || !report.available) && (
        <Typography color="text.secondary">
          No human labels yet. Generate a template with `python -m app.evals label-template
          &lt;eval_run_id&gt;`, hand-fill ~40 answers into backend/app/evals/labels/human_labels.yaml,
          and reload.
        </Typography>
      )}
      {!loading && report?.available && (
        <Stack spacing={2}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Dimension</TableCell>
                <TableCell align="right">n</TableCell>
                <TableCell align="right">Exact %</TableCell>
                <TableCell align="right">Within-1 %</TableCell>
                <TableCell align="right">Cohen&apos;s κ</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {report.dimensions.map((d) => (
                <TableRow key={d.dimension}>
                  <TableCell>{d.dimension.replaceAll("_", " ")}</TableCell>
                  <TableCell align="right">{d.n}</TableCell>
                  <TableCell align="right">{d.exact_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{d.within1_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{d.kappa.toFixed(3)}</TableCell>
                </TableRow>
              ))}
              {report.overall && (
                <TableRow>
                  <TableCell sx={{ fontWeight: 700 }}>overall (pooled)</TableCell>
                  <TableCell align="right">{report.overall.n}</TableCell>
                  <TableCell align="right">{report.overall.exact_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{report.overall.within1_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{report.overall.kappa.toFixed(3)}</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
          <Stack direction="row" spacing={3} flexWrap="wrap" useFlexGap>
            {report.dimensions.map((d) => (
              <Matrix key={d.dimension} dim={d} />
            ))}
          </Stack>
        </Stack>
      )}
    </Paper>
  );
}
