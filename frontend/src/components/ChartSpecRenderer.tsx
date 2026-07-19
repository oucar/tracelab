import { Paper, Typography } from "@mui/material";
import { BarChart } from "@mui/x-charts/BarChart";
import { LineChart } from "@mui/x-charts/LineChart";
import { PieChart } from "@mui/x-charts/PieChart";
import { ScatterChart } from "@mui/x-charts/ScatterChart";
import { chartSpecSchema } from "../lib/chartSpec";

const HEIGHT = 300;

/** Renders a validated ChartSpec with MUI X Charts. Invalid specs render nothing. */
export function ChartSpecRenderer({ spec: raw }: { spec: unknown }) {
  const parsed = chartSpecSchema.safeParse(raw);
  if (!parsed.success) return null;
  const spec = parsed.data;
  const data = spec.data as Record<string, number | string>[];
  const series = spec.y.map((key) => ({ dataKey: key, label: key }));

  let chart: React.ReactNode;
  switch (spec.kind) {
    case "bar":
    case "histogram":
      chart = (
        <BarChart
          dataset={data}
          xAxis={[{ scaleType: "band", dataKey: spec.x, label: spec.x_label || undefined }]}
          yAxis={[{ label: spec.y_label || undefined }]}
          series={series}
          height={HEIGHT}
        />
      );
      break;
    case "line":
      chart = (
        <LineChart
          dataset={data}
          xAxis={[{ scaleType: "point", dataKey: spec.x, label: spec.x_label || undefined }]}
          yAxis={[{ label: spec.y_label || undefined }]}
          series={series}
          height={HEIGHT}
        />
      );
      break;
    case "scatter":
      chart = (
        <ScatterChart
          series={[
            {
              label: spec.y[0],
              data: data.map((row, i) => ({
                id: i,
                x: Number(row[spec.x]),
                y: Number(row[spec.y[0]]),
              })),
            },
          ]}
          xAxis={[{ label: spec.x_label || spec.x }]}
          yAxis={[{ label: spec.y_label || spec.y[0] }]}
          height={HEIGHT}
        />
      );
      break;
    case "pie":
      chart = (
        <PieChart
          series={[
            {
              data: data.map((row, i) => ({
                id: i,
                label: String(row[spec.x]),
                value: Number(row[spec.y[0]]),
              })),
            },
          ]}
          height={HEIGHT}
        />
      );
      break;
  }

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      {spec.title && (
        <Typography variant="subtitle2" gutterBottom>
          {spec.title}
        </Typography>
      )}
      {chart}
    </Paper>
  );
}
