import { useState, useCallback } from 'react';

/**
 * Custom hook for error state management.
 * Manages a single error message string (or null when no error).
 * Form state preservation is handled at the component level —
 * this hook only manages the error display state.
 */
export function useError() {
  const [error, setErrorState] = useState(null);

  /** Set an error message. */
  const setError = useCallback((message) => {
    setErrorState(message);
  }, []);

  /** Clear the current error. */
  const clearError = useCallback(() => {
    setErrorState(null);
  }, []);

  return {
    error,
    setError,
    clearError,
  };
}

export default useError;
