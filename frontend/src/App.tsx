import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Workbench } from "./pages/Workbench";

const theme = createTheme({
  palette: { mode: "dark", primary: { main: "#7aa2f7" }, background: { default: "#0f1115" } },
  typography: { fontFamily: "Inter, system-ui, sans-serif" },
  shape: { borderRadius: 10 },
});

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Workbench />
      </ThemeProvider>
    </QueryClientProvider>
  );
}
