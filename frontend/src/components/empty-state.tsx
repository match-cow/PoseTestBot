import type { LucideIcon } from "lucide-react"

export function EmptyState({ icon: Icon, title, description, action }: { icon: LucideIcon; title: string; description: string; action?: React.ReactNode }) {
  return (
    <div className="grid min-h-52 place-items-center rounded-xl border border-dashed border-border bg-muted/20 p-8 text-center">
      <div className="max-w-md">
        <div className="mx-auto mb-4 grid size-11 place-items-center rounded-full bg-muted"><Icon className="size-5 text-muted-foreground" /></div>
        <h3 className="font-display font-semibold">{title}</h3>
        <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        {action && <div className="mt-4">{action}</div>}
      </div>
    </div>
  )
}
