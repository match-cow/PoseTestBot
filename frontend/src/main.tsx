import React from "react"
import ReactDOM from "react-dom/client"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { HashRouter } from "react-router-dom"
import { App } from "@/app"
import { Toaster } from "@/components/ui/sonner"
import { OperatorProvider } from "@/providers/operator-provider"
import { ThemeProvider } from "@/providers/theme-provider"
import "@/index.css"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
    mutations: { retry: 0 },
  },
})

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <OperatorProvider>
          <HashRouter><App /></HashRouter>
          <Toaster />
        </OperatorProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
