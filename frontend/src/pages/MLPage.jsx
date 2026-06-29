import { useState, useEffect } from 'react'
import {
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { trainModel, getFeatureImportance } from '../services/mlService'
import api from '../services/api'
import { useDatasetContext } from '../contexts/DatasetContext'
import MetricsCard from '../components/MetricsCard'
import ChartRenderer from '../components/ChartRenderer'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'

/**
 * ConfusionMatrixHeatmap — Renders a confusion matrix as a colored grid.
 */
function ConfusionMatrixHeatmap({ matrix }) {
  if (!matrix || matrix.length === 0) return null

  const maxVal = Math.max(...matrix.flat())

  function cellColor(value) {
    const intensity = maxVal > 0 ? value / maxVal : 0
    const r = Math.round(59 + (37 - 59) * intensity)
    const g = Math.round(130 + (99 - 130) * intensity)
    const b = Math.round(246 + (235 - 246) * intensity)
    return `rgb(${r}, ${g}, ${b})`
  }

  return (
    <div className="overflow-auto">
      <div className="inline-block">
        <div className="flex items-end mb-2">
          <div className="w-16" />
          <p className="text-xs font-medium text-gray-500 text-center flex-1">Predicted</p>
        </div>
        {matrix.map((row, rowIdx) => (
          <div key={rowIdx} className="flex">
            {rowIdx === 0 && (
              <div className="w-16 flex items-center justify-center">
                <span className="text-xs font-medium text-gray-500 -rotate-90 whitespace-nowrap">
                  Actual
                </span>
              </div>
            )}
            {rowIdx !== 0 && <div className="w-16" />}
            {row.map((value, colIdx) => (
              <div
                key={colIdx}
                className="w-16 h-14 flex items-center justify-center text-sm font-mono font-medium border border-white rounded"
                style={{ backgroundColor: cellColor(value), color: value > maxVal * 0.6 ? '#fff' : '#1f2937' }}
                title={`Actual: ${rowIdx}, Predicted: ${colIdx}, Count: ${value}`}
              >
                {value}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/**
 * ActualVsPredictedScatter — Scatter plot with diagonal reference line for regression.
 */
function ActualVsPredictedScatter({ data, diagonalLine }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" name="Actual" type="number" label={{ value: 'Actual', position: 'bottom', offset: 0 }} />
        <YAxis dataKey="y" name="Predicted" type="number" label={{ value: 'Predicted', angle: -90, position: 'insideLeft' }} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Legend />
        <Scatter name="Predictions" data={data} fill="#8884d8" />
        <Scatter name="Perfect Prediction" data={diagonalLine} fill="none" line={{ stroke: '#ef4444', strokeWidth: 2, strokeDasharray: '5 5' }} shape={() => null} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

/**
 * FeatureImportanceChart — Horizontal bar chart for feature importance scores.
 */
function FeatureImportanceChart({ data }) {
  if (!data || data.length === 0) return null

  return (
    <ResponsiveContainer width="100%" height={Math.max(300, data.length * 40)}>
      <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 100]} label={{ value: 'Importance (%)', position: 'bottom', offset: 0 }} />
        <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
        <Bar dataKey="importance" fill="#8884d8" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

/**
 * Algorithm options per task type.
 */
const CLASSIFICATION_ALGORITHMS = [
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'decision_tree', label: 'Decision Tree' },
  { value: 'random_forest', label: 'Random Forest' },
]

const REGRESSION_ALGORITHMS = [
  { value: 'linear_regression', label: 'Linear Regression' },
  { value: 'decision_tree', label: 'Decision Tree' },
  { value: 'random_forest', label: 'Random Forest' },
]

/**
 * Tree-based algorithms that support feature importance.
 */
const TREE_BASED = ['decision_tree', 'random_forest']

/**
 * MLPage — Model training configuration and results display.
 *
 * Provides task type selection, target column dropdown, algorithm dropdown,
 * train button, and displays metrics + charts after training.
 * (Requirements 10.1–10.6, 11.1–11.6, 12.1–12.4)
 */
function MLPage() {
  // Dataset context
  const { filename: contextFilename } = useDatasetContext()

  // Dataset metadata
  const [columns, setColumns] = useState([])
  const [dtypes, setDtypes] = useState({})
  const [filename, setFilename] = useState(null)

  // Configuration state
  const [taskType, setTaskType] = useState('classification')
  const [targetColumn, setTargetColumn] = useState('')
  const [algorithm, setAlgorithm] = useState('')

  // Results state
  const [results, setResults] = useState(null)
  const [featureImportance, setFeatureImportance] = useState(null)

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [summaryLoading, setSummaryLoading] = useState(true)

  // Validation errors state: { fieldName: "error message" }
  const [validationErrors, setValidationErrors] = useState({})

  // Fetch dataset summary on mount
  useEffect(() => {
    if (!contextFilename) {
      setSummaryLoading(false)
      return
    }
    setFilename(contextFilename)

    api
      .get('/summary', { params: { filename: contextFilename } })
      .then((data) => {
        setColumns(data.columns || [])
        setDtypes(data.dtypes || {})
      })
      .catch((err) => {
        setError(err.message || 'Failed to load dataset summary')
      })
      .finally(() => setSummaryLoading(false))
  }, [contextFilename])

  // Reset target and algorithm when task type changes
  useEffect(() => {
    setTargetColumn('')
    setAlgorithm('')
  }, [taskType])

  /**
   * Filter columns based on task type:
   * - classification: categorical columns (object, category, bool)
   * - regression: numerical columns (int64, float64, int32, float32)
   */
  const filteredTargetColumns = columns.filter((col) => {
    const dtype = (dtypes[col] || '').toLowerCase()
    if (taskType === 'classification') {
      return dtype.includes('object') || dtype.includes('category') || dtype.includes('bool')
    }
    return dtype.includes('int') || dtype.includes('float')
  })

  const algorithmOptions =
    taskType === 'classification' ? CLASSIFICATION_ALGORITHMS : REGRESSION_ALGORITHMS

  /**
   * Handle model training with per-field validation.
   */
  async function handleTrain() {
    // Client-side validation before backend request
    const errors = {}

    if (!targetColumn) {
      errors.targetColumn = 'Please select a target column.'
    }
    if (!algorithm) {
      errors.algorithm = 'Please select an algorithm.'
    }

    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors)
      return
    }

    // Clear validation errors on successful validation
    setValidationErrors({})

    if (!filename) return

    setLoading(true)
    setError(null)
    setResults(null)
    setFeatureImportance(null)

    try {
      const data = await trainModel(filename, targetColumn, algorithm, taskType)
      setResults(data)

      // Fetch feature importance for tree-based models
      if (TREE_BASED.includes(algorithm)) {
        try {
          const fiData = await getFeatureImportance(filename)
          setFeatureImportance(fiData)
        } catch {
          // Feature importance is optional; don't block on failure
          setFeatureImportance({ available: false, message: 'Could not load feature importance.' })
        }
      } else {
        setFeatureImportance({
          available: false,
          message: 'Feature importance is only available for tree-based models (Decision Tree, Random Forest).',
        })
      }
    } catch (err) {
      setError(err.message || 'Model training failed. Please check your configuration and try again.')
    } finally {
      setLoading(false)
    }
  }

  /**
   * Build scatter data for actual vs predicted chart.
   */
  function buildActualVsPredicted() {
    if (!results?.predictions) return []
    const { actual, predicted } = results.predictions
    return actual.map((a, i) => ({ x: a, y: predicted[i] }))
  }

  /**
   * Build diagonal reference line data for regression scatter.
   */
  function buildDiagonalLine() {
    if (!results?.predictions) return []
    const { actual } = results.predictions
    const min = Math.min(...actual)
    const max = Math.max(...actual)
    return [
      { x: min, y: min },
      { x: max, y: max },
    ]
  }

  /**
   * Build feature importance chart data sorted descending.
   */
  function buildFeatureImportanceData() {
    if (!featureImportance?.features) return []
    return [...featureImportance.features]
      .sort((a, b) => b.importance - a.importance)
      .map((f) => ({ name: f.name, importance: Number((f.importance * 100).toFixed(2)) }))
  }

  if (summaryLoading) {
    return (
      <div className="p-6">
        <LoadingSpinner loading={true} message="Loading dataset information..." />
      </div>
    )
  }

  if (!filename) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-800">Machine Learning</h1>
        <p className="mt-4 text-gray-600">
          No dataset loaded. Please upload a CSV file first.
        </p>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-800">Machine Learning</h1>
        <p className="mt-2 text-gray-600">
          Train classification and regression models on your dataset.
        </p>
      </div>

      {/* Error display */}
      {error && (
        <ErrorAlert error={error} onDismiss={() => setError(null)} operation="Model Training" />
      )}

      {/* Configuration section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm space-y-6">
        <h2 className="text-xl font-semibold text-gray-800">Model Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Task type selector */}
          <div>
            <label htmlFor="task-type" className="block text-sm font-medium text-gray-700 mb-1">
              Task Type
            </label>
            <select
              id="task-type"
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>

          {/* Target column dropdown */}
          <div>
            <label htmlFor="target-column" className="block text-sm font-medium text-gray-700 mb-1">
              Target Column
            </label>
            <select
              id="target-column"
              value={targetColumn}
              onChange={(e) => {
                setTargetColumn(e.target.value)
                setValidationErrors((prev) => { const { targetColumn: _, ...rest } = prev; return rest })
              }}
              className={`w-full rounded-md border px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 ${
                validationErrors.targetColumn ? 'border-red-500 ring-1 ring-red-500' : 'border-gray-300'
              }`}
              aria-invalid={!!validationErrors.targetColumn}
              aria-describedby={validationErrors.targetColumn ? 'target-column-error' : undefined}
            >
              <option value="">Select target column...</option>
              {filteredTargetColumns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
            {validationErrors.targetColumn && (
              <p id="target-column-error" className="mt-1 text-xs text-red-600 flex items-center gap-1">
                <svg className="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                {validationErrors.targetColumn}
              </p>
            )}
            {filteredTargetColumns.length === 0 && (
              <p className="mt-1 text-xs text-amber-600">
                No {taskType === 'classification' ? 'categorical' : 'numerical'} columns found.
              </p>
            )}
          </div>

          {/* Algorithm dropdown */}
          <div>
            <label htmlFor="algorithm" className="block text-sm font-medium text-gray-700 mb-1">
              Algorithm
            </label>
            <select
              id="algorithm"
              value={algorithm}
              onChange={(e) => {
                setAlgorithm(e.target.value)
                setValidationErrors((prev) => { const { algorithm: _, ...rest } = prev; return rest })
              }}
              className={`w-full rounded-md border px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 ${
                validationErrors.algorithm ? 'border-red-500 ring-1 ring-red-500' : 'border-gray-300'
              }`}
              aria-invalid={!!validationErrors.algorithm}
              aria-describedby={validationErrors.algorithm ? 'algorithm-error' : undefined}
            >
              <option value="">Select algorithm...</option>
              {algorithmOptions.map((algo) => (
                <option key={algo.value} value={algo.value}>
                  {algo.label}
                </option>
              ))}
            </select>
            {validationErrors.algorithm && (
              <p id="algorithm-error" className="mt-1 text-xs text-red-600 flex items-center gap-1">
                <svg className="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                {validationErrors.algorithm}
              </p>
            )}
          </div>
        </div>

        {/* Train button */}
        <div>
          <button
            onClick={handleTrain}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Training...' : 'Train Model'}
          </button>
        </div>

        <LoadingSpinner loading={loading} message="Training model, please wait..." />
      </div>

      {/* Results section */}
      {results && (
        <div className="space-y-8">
          {/* Metrics cards */}
          <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Model Metrics</h2>

            {taskType === 'classification' && results.metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricsCard
                  label="Accuracy"
                  value={(results.metrics.accuracy * 100).toFixed(2)}
                  unit="%"
                  description="Overall correctness"
                />
                <MetricsCard
                  label="Precision"
                  value={(results.metrics.precision * 100).toFixed(2)}
                  unit="%"
                  description="Positive prediction accuracy"
                />
                <MetricsCard
                  label="Recall"
                  value={(results.metrics.recall * 100).toFixed(2)}
                  unit="%"
                  description="True positive rate"
                />
                <MetricsCard
                  label="F1 Score"
                  value={(results.metrics.f1_score * 100).toFixed(2)}
                  unit="%"
                  description="Harmonic mean of precision and recall"
                />
              </div>
            )}

            {taskType === 'regression' && results.metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricsCard
                  label="R² Score"
                  value={results.metrics.r2_score?.toFixed(4)}
                  description="Variance explained"
                />
                <MetricsCard
                  label="MAE"
                  value={results.metrics.mae?.toFixed(4)}
                  description="Mean Absolute Error"
                />
                <MetricsCard
                  label="MSE"
                  value={results.metrics.mse?.toFixed(4)}
                  description="Mean Squared Error"
                />
                <MetricsCard
                  label="RMSE"
                  value={results.metrics.rmse?.toFixed(4)}
                  description="Root Mean Squared Error"
                />
              </div>
            )}
          </div>

          {/* Confusion Matrix (classification only) */}
          {taskType === 'classification' && results.confusion_matrix && (
            <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Confusion Matrix</h2>
              <ConfusionMatrixHeatmap matrix={results.confusion_matrix} />
            </div>
          )}

          {/* Actual vs Predicted scatter plot */}
          {results.predictions && (
            <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Actual vs Predicted</h2>
              {taskType === 'regression' ? (
                <ActualVsPredictedScatter
                  data={buildActualVsPredicted()}
                  diagonalLine={buildDiagonalLine()}
                />
              ) : (
                <ChartRenderer
                  type="scatter"
                  data={buildActualVsPredicted()}
                  title=""
                />
              )}
            </div>
          )}

          {/* Feature Importance */}
          <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Feature Importance</h2>
            {featureImportance?.available === false ? (
              <div className="flex items-center gap-2 text-gray-500 py-4">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
                </svg>
                <p className="text-sm">{featureImportance.message}</p>
              </div>
            ) : featureImportance?.features ? (
              <FeatureImportanceChart data={buildFeatureImportanceData()} />
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}

export default MLPage
