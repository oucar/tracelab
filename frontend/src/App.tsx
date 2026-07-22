import { AppBar, Box, Stack, ThemeProvider, Toolbar, Tooltip, Typography } from "@mui/material";
import { CssBaseline } from "@mui/material";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import {
  BrowserRouter,
  Link as RouterLink,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import { getConfig } from "./lib/api";
import { EvalsScreen } from "./pages/EvalsScreen";
import { RunsDashboard } from "./pages/RunsDashboard";
import { RunView } from "./pages/RunView";
import { Workbench } from "./pages/Workbench";
import { MONO, theme, tokens } from "./theme";

const queryClient = new QueryClient();

/** A small node-graph glyph — the agent pipeline, rendered as the mark. */
function BrandMark() {
  return (
    <Box
      component="svg"
      viewBox="0 0 28 28"
      sx={{ width: 22, height: 22, display: "block", flexShrink: 0 }}
      aria-hidden
    >
      <line x1="6" y1="14" x2="14" y2="7" stroke={tokens.accent} strokeWidth="1.5" opacity="0.55" />
      <line x1="6" y1="14" x2="14" y2="21" stroke={tokens.accent} strokeWidth="1.5" opacity="0.55" />
      <line x1="14" y1="7" x2="22" y2="14" stroke={tokens.accent} strokeWidth="1.5" opacity="0.55" />
      <line x1="14" y1="21" x2="22" y2="14" stroke={tokens.accent} strokeWidth="1.5" opacity="0.55" />
      <circle cx="6" cy="14" r="2.6" fill={tokens.accentBright} />
      <circle cx="14" cy="7" r="2.6" fill={tokens.accent} />
      <circle cx="14" cy="21" r="2.6" fill="#22d3ee" />
      <circle cx="22" cy="14" r="2.6" fill="#e879f9" />
    </Box>
  );
}

function NavLink({ to, label }: { to: string; label: string }) {
  const { pathname } = useLocation();
  const active = to === "/" ? pathname === "/" : pathname.startsWith(to);
  return (
    <Box
      component={RouterLink}
      to={to}
      sx={{
        px: 1.25,
        py: 0.5,
        borderRadius: 2,
        fontSize: "0.82rem",
        fontWeight: 600,
        textDecoration: "none",
        color: active ? "text.primary" : "text.secondary",
        bgcolor: active ? "action.selected" : "transparent",
        transition: "color 150ms ease, background-color 150ms ease",
        "&:hover": { color: "text.primary", bgcolor: active ? "action.selected" : "action.hover" },
      }}
    >
      {label}
    </Box>
  );
}

function BudgetMeter() {
  const { data } = useQuery({ queryKey: ["config"], queryFn: getConfig, refetchInterval: 60_000 });
  if (!data) return null;
  const pct = Math.min(100, (data.spent_today / data.daily_budget_usd) * 100);
  const near = pct > 80;
  return (
    <Stack direction="row" spacing={1.25} alignItems="center">
      {data.cheap_mode && (
        <Tooltip title="Mini models everywhere — cheap day-to-day mode">
          <Box
            sx={{
              px: 1,
              py: "3px",
              borderRadius: 1.5,
              border: "1px solid",
              borderColor: "warning.main",
              color: "warning.main",
              fontFamily: MONO,
              fontSize: "0.66rem",
              fontWeight: 600,
              letterSpacing: "0.04em",
            }}
          >
            CHEAP
          </Box>
        </Tooltip>
      )}
      <Tooltip title="Daily API spend against the budget cap">
        <Stack spacing={0.5} sx={{ minWidth: 96 }}>
          <Typography
            sx={{ fontFamily: MONO, fontSize: "0.66rem", color: "text.secondary", lineHeight: 1 }}
          >
            ${data.spent_today.toFixed(2)}
            <Box component="span" sx={{ color: "text.disabled" }}>
              {" "}
              / ${data.daily_budget_usd.toFixed(0)}
            </Box>
          </Typography>
          <Box sx={{ height: 3, borderRadius: 999, bgcolor: "var(--surface-3)", overflow: "hidden" }}>
            <Box
              sx={{
                height: "100%",
                width: `${pct}%`,
                borderRadius: 999,
                bgcolor: near ? "warning.main" : "primary.main",
                transition: "width 400ms var(--ease-out)",
              }}
            />
          </Box>
        </Stack>
      </Tooltip>
    </Stack>
  );
}

function Nav() {
  return (
    <AppBar
      position="sticky"
      color="transparent"
      elevation={0}
      sx={{
        borderBottom: "1px solid",
        borderColor: "divider",
        backdropFilter: "blur(12px)",
        backgroundColor: "rgba(10,12,16,0.72)",
      }}
    >
      <Toolbar variant="dense" sx={{ gap: 1, minHeight: 52 }}>
        <Box
          component={RouterLink}
          to="/"
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            textDecoration: "none",
            color: "text.primary",
            mr: 1.5,
          }}
        >
          <BrandMark />
          <Typography sx={{ fontWeight: 750, fontSize: "1rem", letterSpacing: "-0.02em" }}>
            tracelab
          </Typography>
        </Box>
        <NavLink to="/" label="Workbench" />
        <NavLink to="/runs" label="Runs" />
        <NavLink to="/evals" label="Evals" />
        <Box sx={{ flexGrow: 1 }} />
        <BudgetMeter />
      </Toolbar>
    </AppBar>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <Nav />
          <Routes>
            <Route path="/" element={<Workbench />} />
            <Route path="/runs" element={<RunsDashboard />} />
            <Route path="/runs/:runId" element={<RunView />} />
            <Route path="/evals" element={<EvalsScreen />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
