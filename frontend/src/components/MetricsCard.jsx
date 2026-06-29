/**
 * MetricsCard — Numeric metric display card for model evaluation results.
 *
 * Displays a single metric value with its label and optional unit/description.
 * Used on the ML page to present classification and regression metrics
 * such as accuracy, R² score, MAE, etc.
 * (Requirements 10.3, 11.3)
 *
 * Props:
 *   label (string) — the metric name (e.g., "Accuracy", "R² Score", "MAE")
 *   value (number | string) — the metric value
 *   unit (string, optional) — unit suffix appended to value (e.g., "%")
 *   description (string, optional) — brief description shown below the value
 */
function MetricsCard({ label, value, unit, description }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <p className="text-sm font-medium text-gray-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-gray-900">
        {value}
        {unit && <span className="text-lg font-semibold text-gray-600">{unit}</span>}
      </p>
      {description && (
        <p className="mt-1 text-xs text-gray-400">{description}</p>
      )}
    </div>
  )
}

export default MetricsCard
