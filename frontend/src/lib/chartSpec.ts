import { z } from "zod";

/** Zod mirror of the backend ChartSpec (backend/app/runtime/chartspec.py). */
export const chartSpecSchema = z.object({
  kind: z.enum(["line", "bar", "scatter", "pie", "histogram"]),
  title: z.string().default(""),
  x: z.string(),
  y: z.array(z.string()).min(1),
  data: z.array(z.record(z.string(), z.unknown())).min(1).max(500),
  x_label: z.string().default(""),
  y_label: z.string().default(""),
  source_columns: z.array(z.string()).default([]),
});

export type ChartSpec = z.infer<typeof chartSpecSchema>;
