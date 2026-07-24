import { createTheme } from "@mui/material/styles";

/**
 * tracelab design system — "trace-native".
 * A precise, engineered dark surface. Inter for UI, JetBrains Mono for data.
 * The accent (periwinkle) is the product's existing identity, kept and systematized.
 */

export const MONO = "'JetBrains Mono Variable', ui-monospace, 'SF Mono', Menlo, monospace";

// Neutral ramp — near-black cooled slightly toward the accent hue.
const c = {
  bg: "#0a0c10", // app background
  surface: "#111318", // panels
  surface2: "#171a21", // elevated / inputs / hover
  surface3: "#1d212a", // pressed / selected
  border: "#23262e",
  borderStrong: "#32363f",
  ink0: "#f5f7fb", // headings
  ink1: "#e4e7ee", // primary text
  ink2: "#a7aec0", // secondary (AA on surfaces)
  ink3: "#7d8496", // faint labels
  accent: "#7aa2f7",
  accentBright: "#a3c0fc",
  accentDim: "#3a4a74",
};

/** Per-agent identity colors — a designed cool→warm spectrum, distinct from status hues. */
export const AGENT: Record<string, string> = {
  system: "#8b93a7",
  router: "#a78bfa", // violet — the classifier at the entrance
  planner: "#7aa2f7", // blue — the orchestrator (brand family)
  analyst: "#22d3ee", // cyan — the workers
  critic: "#fb923c", // orange — scrutiny
  composer: "#e879f9", // fuchsia — the writer
};

export const STATUS = {
  verified: "#4ade80",
  unverified: "#fbbf24",
  error: "#f87171",
  active: c.accent,
  pending: c.ink3,
};

export const tokens = c;

export const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: c.accent, light: c.accentBright, dark: c.accentDim, contrastText: "#0a0c10" },
    success: { main: STATUS.verified },
    warning: { main: STATUS.unverified },
    error: { main: STATUS.error },
    info: { main: AGENT.analyst },
    background: { default: c.bg, paper: c.surface },
    text: { primary: c.ink1, secondary: c.ink2, disabled: c.ink3 },
    divider: c.border,
  },
  shape: { borderRadius: 12 },
  typography: {
    fontFamily: "'Inter Variable', Inter, system-ui, -apple-system, sans-serif",
    h4: { fontSize: "1.5rem", fontWeight: 700, letterSpacing: "-0.02em" },
    h5: { fontSize: "1.25rem", fontWeight: 700, letterSpacing: "-0.015em" },
    h6: { fontSize: "1.05rem", fontWeight: 650, letterSpacing: "-0.01em" },
    subtitle1: { fontSize: "0.95rem", fontWeight: 600 },
    subtitle2: {
      fontSize: "0.7rem",
      fontWeight: 600,
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      color: c.ink3,
    },
    body1: { fontSize: "0.9rem", lineHeight: 1.55 },
    body2: { fontSize: "0.825rem", lineHeight: 1.5 },
    caption: { fontSize: "0.72rem", letterSpacing: "0.01em" },
    button: { textTransform: "none", fontWeight: 600, letterSpacing: 0 },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        // Kill MUI's dark-mode elevation gradient overlay everywhere.
        body: { backgroundColor: c.bg },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: "none" },
        outlined: { borderColor: c.border, backgroundColor: c.surface },
      },
    },
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: {
          borderRadius: 10,
          transition: "background-color 150ms ease, color 150ms ease, border-color 150ms ease",
        },
        contained: {
          boxShadow: "none",
          "&:hover": { boxShadow: `0 0 0 1px ${c.accentBright}55, 0 6px 20px -8px ${c.accent}88` },
        },
        text: { color: c.ink2, "&:hover": { color: c.ink0, backgroundColor: c.surface2 } },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { fontWeight: 600, fontSize: "0.72rem", height: 24 },
        outlined: { borderColor: c.borderStrong },
        label: { paddingLeft: 9, paddingRight: 9 },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          backgroundColor: c.surface2,
          borderRadius: 10,
          "& fieldset": { borderColor: c.border },
          "&:hover fieldset": { borderColor: c.borderStrong },
          "&.Mui-focused fieldset": { borderColor: c.accent, borderWidth: 1 },
          "&.Mui-focused": { boxShadow: `0 0 0 3px ${c.accent}22` },
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: c.surface3,
          border: `1px solid ${c.borderStrong}`,
          color: c.ink1,
          fontSize: "0.72rem",
          fontWeight: 500,
          borderRadius: 8,
          padding: "6px 10px",
          boxShadow: "0 8px 24px -8px rgba(0,0,0,0.6)",
        },
        arrow: { color: c.surface3 },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: { borderRadius: 12, border: `1px solid ${c.border}`, alignItems: "flex-start" },
        standardSuccess: { backgroundColor: "#4ade8014", color: c.ink1 },
        standardWarning: { backgroundColor: "#fbbf2414", color: c.ink1 },
        standardError: { backgroundColor: "#f8717114", color: c.ink1 },
      },
    },
    MuiLinearProgress: {
      styleOverrides: { root: { borderRadius: 999, backgroundColor: c.surface3 } },
    },
    MuiDivider: { styleOverrides: { root: { borderColor: c.border } } },
  },
});
