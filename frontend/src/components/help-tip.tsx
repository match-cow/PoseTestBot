import { CircleHelp } from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export interface HelpTipProps {
  label: string
  children: React.ReactNode
  className?: string
  contentClassName?: string
}

/** Supplemental, keyboard-accessible help. Required instructions must remain visible. */
export function HelpTip({ label, children, className, contentClassName }: HelpTipProps) {
  return <Tooltip>
    <TooltipTrigger asChild>
      <button
        type="button"
        aria-label={`About ${label}`}
        className={cn("inline-grid size-5 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60", className)}
      >
        <CircleHelp aria-hidden="true" className="size-3.5" />
      </button>
    </TooltipTrigger>
    <TooltipContent sideOffset={6} className={cn("max-w-80 text-pretty px-3 py-2 text-xs leading-relaxed", contentClassName)}>
      {children}
    </TooltipContent>
  </Tooltip>
}
