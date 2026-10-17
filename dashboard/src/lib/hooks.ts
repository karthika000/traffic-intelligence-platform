import { useEffect, useRef } from "react";

/** Calls `fn` at most once per `ms` while `value` keeps changing. */
export function useThrottledEffect(value: unknown, ms: number, fn: () => void) {
  const last = useRef(0);
  const first = useRef(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const wait = Math.max(0, last.current + ms - Date.now());
    const t = setTimeout(() => {
      last.current = Date.now();
      fnRef.current();
    }, wait);
    return () => clearTimeout(t);
  }, [value, ms]);
}
