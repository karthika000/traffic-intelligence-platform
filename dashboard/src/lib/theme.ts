import { useStore } from "zustand";
import { createStore } from "zustand/vanilla";

export type Theme = "dark" | "light";
const KEY = "tip.theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* storage unavailable */
  }
  return "dark"; // operations-centre default
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export const themeStore = createStore<{ theme: Theme; toggle: () => void }>()((set, get) => ({
  theme: initialTheme(),
  toggle: () => {
    const theme: Theme = get().theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* not persisted */
    }
    applyTheme(theme);
    set({ theme });
  },
}));

export const useTheme = () => useStore(themeStore, (s) => s.theme);
