import { lazy, Suspense } from "react"
import { Navigate, Route, Routes } from "react-router-dom"
import { AppShell } from "@/components/app-shell"

const DashboardPage = lazy(() => import("@/features/dashboard/dashboard-page").then((module) => ({ default: module.DashboardPage })))
const DevicesPage = lazy(() => import("@/features/devices/devices-page").then((module) => ({ default: module.DevicesPage })))
const WorkflowPage = lazy(() => import("@/features/workflow/workflow-page").then((module) => ({ default: module.WorkflowPage })))
const ArtifactsPage = lazy(() => import("@/features/artifacts/artifacts-page").then((module) => ({ default: module.ArtifactsPage })))
const JobsPage = lazy(() => import("@/features/jobs/jobs-page").then((module) => ({ default: module.JobsPage })))
const CellPage = lazy(() => import("@/features/cell/cell-page").then((module) => ({ default: module.CellPage })))

function Page({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<div className="grid min-h-[60vh] place-items-center text-sm text-muted-foreground">Loading view…</div>}>{children}</Suspense>
}

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Page><DashboardPage /></Page>} />
        <Route path="/devices" element={<Page><DevicesPage /></Page>} />
        <Route path="/cell" element={<Page><CellPage /></Page>} />
        <Route path="/workflow/:phase" element={<Page><WorkflowPage /></Page>} />
        <Route path="/artifacts" element={<Page><ArtifactsPage /></Page>} />
        <Route path="/jobs" element={<Page><JobsPage /></Page>} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  )
}
