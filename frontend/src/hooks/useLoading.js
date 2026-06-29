import { useState, useCallback, useRef } from 'react';

/**
 * Custom hook for loading state with 300ms debounce.
 * Prevents spinner flash for fast operations by delaying the visible loading state.
 *
 * - `isLoading`: raw state, true immediately when startLoading is called
 * - `showLoading`: debounced state, true only after 300ms of loading
 */
export function useLoading() {
  const [isLoading, setIsLoading] = useState(false);
  const [showLoading, setShowLoading] = useState(false);
  const timerRef = useRef(null);

  /** Start loading. Sets isLoading immediately; sets showLoading after 300ms. */
  const startLoading = useCallback(() => {
    setIsLoading(true);
    timerRef.current = setTimeout(() => {
      setShowLoading(true);
    }, 300);
  }, []);

  /** Stop loading. Clears the debounce timer and resets both states. */
  const stopLoading = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setIsLoading(false);
    setShowLoading(false);
  }, []);

  return {
    isLoading,
    showLoading,
    startLoading,
    stopLoading,
  };
}

export default useLoading;
