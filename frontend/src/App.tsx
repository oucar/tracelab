import { AppBar, Box, Button, Chip, CssBaseline, Stack, ThemeProvider, Toolbar, Typography, createTheme } from "@mui/material";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { BrowserRouter, Link as RouterLink, Route, Routes } from "react-router-dom";
import { getConfig } from "./lib/api";
import { RunsDashboard } from "./pages/RunsDashboard";
import { RunView } from "./pages/RunView";
import { Workbench } from "./pages/Workbench";

const theme = createTheme({
  palette: { mode: "dark", primary: { main: "#7aa2f7" }, background: { default: "#0f1115" } },
  typography: { fontFamily: "Inter, system-ui, sans-serif" },
  shape: { borderRadius: 10 },
});

const queryClient = new QueryClient();

function ConfigChips() {
  const { data } = useQuery({ queryKey: ["config"], queryFn: getConfig, refetchInterval: 60_000 });
  if (!data) return null;
  return (
    <Stack direction="row" spacing={1}>
      {data.cheap_mode && (
        <Chip size="small" color="warning" variant="outlined" label="CHEAP_MODE" />
      )}
      <Chip
        size="small"
        variant="outlined"
        label={`$${data.spent_today.toFixed(2)} / $${data.daily_budget_usd.toFixed(2)} today`}
      />
    </Stack>
  );
}

function Nav() {
  return (
    <AppBar position="sticky" color="transparent" elevation={0} sx={{ borderBottom: 1, borderColor: "divider", backdropFilter: "blur(8px)" }}>
      <Toolbar variant="dense" sx={{ gap: 2 }}>
        <Typography
          variant="h6"
          fontWeight={700}
          component={RouterLink}
          to="/"
          sx={{ color: "inherit", textDecoration: "none" }}
        >
          tracelab
        </Typography>
        <Button size="small" color="inherit" component={RouterLink} to="/runs">
          Runs
        </Button>
        <Box sx={{ flexGrow: 1 }} />
        <ConfigChips />
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
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
