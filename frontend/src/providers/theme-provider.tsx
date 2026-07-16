import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react"

export type Theme = "light" | "dark"

const ThemeContext = createContext<{ theme: Theme; setTheme: (theme: Theme) => void } | null>(null)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [hasExplicitTheme, setHasExplicitTheme] = useState(() => {
    const stored = localStorage.getItem("posetestbot.theme")
    return stored === "light" || stored === "dark"
  })
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem("posetestbot.theme")
    if (stored === "light" || stored === "dark") return stored
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
  })

  useEffect(() => {
    const root = document.documentElement
    root.classList.remove("light", "dark")
    root.classList.add(theme)
    root.dataset.theme = theme
    root.style.colorScheme = theme
  }, [theme])

  useEffect(() => {
    if (hasExplicitTheme) return
    localStorage.removeItem("posetestbot.theme")
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const applySystemTheme = (event: MediaQueryListEvent) => setThemeState(event.matches ? "dark" : "light")
    media.addEventListener("change", applySystemTheme)
    return () => media.removeEventListener("change", applySystemTheme)
  }, [hasExplicitTheme])

  const setTheme = useCallback((nextTheme: Theme) => {
    localStorage.setItem("posetestbot.theme", nextTheme)
    setHasExplicitTheme(true)
    setThemeState(nextTheme)
  }, [])

  const value = useMemo(() => ({ theme, setTheme }), [theme, setTheme])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const value = useContext(ThemeContext)
  if (!value) throw new Error("useTheme must be used inside ThemeProvider")
  return value
}
