/**
 * LoadingSpinner — Non-blocking loading indicator.
 *
 * Displays an inline spinning indicator with optional context message.
 * Designed to be shown within 300ms of operation start (Requirement 15.4)
 * and removed when the operation completes (Requirement 15.5).
 *
 * Props:
 *   loading (boolean) — whether to show the spinner
 *   message (string, optional) — context text displayed beside the spinner
 */
function LoadingSpinner({ loading, message }) {
  if (!loading) return null

  return (
    <div className="flex items-center gap-3 py-3" role="status" aria-live="polite">
      <svg
        className="h-5 w-5 animate-spin text-blue-600"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      {message && (
        <span className="text-sm text-gray-600">{message}</span>
      )}
    </div>
  )
}

export default LoadingSpinner
