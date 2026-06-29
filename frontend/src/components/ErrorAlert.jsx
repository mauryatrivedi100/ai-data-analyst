/**
 * ErrorAlert — Dismissible error message display with operation context.
 *
 * Shows a styled alert with the error message and an optional operation
 * context label. Includes a dismiss button that calls onDismiss.
 * (Requirements 15.6, 16.3)
 *
 * Props:
 *   error (string | { message: string }) — the error to display
 *   onDismiss (function) — callback when the user closes the alert
 *   operation (string, optional) — context label for which operation failed
 */
function ErrorAlert({ error, onDismiss, operation }) {
  if (!error) return null

  const message = typeof error === 'string' ? error : error.message

  return (
    <div
      className="relative flex items-start gap-3 rounded-md border border-red-300 bg-red-50 p-4"
      role="alert"
      aria-live="assertive"
    >
      {/* Warning icon */}
      <svg
        className="h-5 w-5 shrink-0 text-red-500 mt-0.5"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 20 20"
        fill="currentColor"
        aria-hidden="true"
      >
        <path
          fillRule="evenodd"
          d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z"
          clipRule="evenodd"
        />
      </svg>

      {/* Message content */}
      <div className="flex-1 text-sm">
        {operation && (
          <p className="font-medium text-red-800 mb-1">
            {operation} failed
          </p>
        )}
        <p className="text-red-700">{message}</p>
      </div>

      {/* Dismiss button */}
      <button
        type="button"
        onClick={onDismiss}
        className="shrink-0 rounded p-1 text-red-400 hover:text-red-600 hover:bg-red-100 transition-colors"
        aria-label="Dismiss error"
      >
        <svg
          className="h-4 w-4"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>
    </div>
  )
}

export default ErrorAlert
