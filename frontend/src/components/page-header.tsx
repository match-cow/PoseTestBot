export function PageHeader({ eyebrow, title, description, actions }: { eyebrow: string; title: string; description: string; actions?: React.ReactNode }) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div className="max-w-3xl">
        <p className="mb-1.5 text-[10px] font-bold uppercase tracking-[0.18em] text-primary-strong">{eyebrow}</p>
        <h1 className="font-display text-[26px] font-semibold leading-tight tracking-tight">{title}</h1>
        <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">{description}</p>
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  )
}
