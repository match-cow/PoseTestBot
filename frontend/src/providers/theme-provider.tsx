import { createContext, useContext, useEffect, useMemo, useState } from "react"

export type Theme = "light" | "dark" | "system"

const ThemeContext = createContext<{ theme: Theme; setTheme: (theme: Theme) => void } | null>(null)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem("posetestbot.theme")
    return stored === "light" || stored === "dark" || stored === "system" ? stored : "system"
  })

  useEffect(() => {
    const root = document.documentElement
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const apply = () => {
      root.classList.remove("light", "dark")
      root.classList.add(theme === "system" ? (media.matches ? "dark" : "light") : theme)
      root.style.colorScheme = theme === "system" ? (media.matches ? "dark" : "light") : theme
    }
    apply()
    localStorage.setItem("posetestbot.theme", theme)
    media.addEventListener("change", apply)
    return () => media.removeEventListener("change", apply)
  }, [theme])

  const value = useMemo(() => ({ theme, setTheme }), [theme])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const value = useContext(ThemeContext)
  if (!value) throw new Error("useTheme must be used inside ThemeProvider")
  return value
}
