import * as React from "react"
import { cn } from "@/lib/utils"

export function Input({ className, type, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input type={type} className={cn("flex h-[34px] w-full rounded-[7px] border border-input bg-card px-3 py-1 text-xs transition-colors duration-150 placeholder:text-muted-foreground hover:border-foreground/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/55 focus-visible:ring-offset-1 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50", className)} {...props} />
}
