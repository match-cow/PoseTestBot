const STORAGE_KEY = "posetestbot.selectedSensors"

export function loadSelectedSensorKeys(): Set<string> {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]")
    if (!Array.isArray(value)) return new Set()
    return new Set(value.filter((item): item is string => typeof item === "string"))
  } catch {
    return new Set()
  }
}

export function saveSelectedSensorKeys(keys: Set<string>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...keys].sort()))
}
