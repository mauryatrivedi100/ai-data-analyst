import { useState, useEffect, useCallback } from 'react'
import api from '../services/api'
import { useDatasetContext } from '../contexts/DatasetContext'
import { getVisualization } from '../services/edaService'
import ChartRenderer from '../components/ChartRenderer'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'
import { useLoading } from '../hooks/useLoading'
import { useError } from '../hooks/useError'

/**
 * Chart type definitions with axis requirements and column type filters.
 * Each entry defines what axes are needed and which column types are valid.
 */
const CHART_TYPES = [
  { value: 'histogram', label: 'Histogram' },
  { value: 'scatter', label: 'Scatter Plot' },
  { value: 'line', label: 'Line Chart' },
  { value: 'bar', label: 'Bar Chart' },
  { value: 'pie', label: 'Pie Chart' },
  { value: 'box', label: 'Box Plot' },
  { value: 'heatmap', label: 'Heatmap' },
]

/**
 * Returns axis configuration for a given chart type.
 * - needsX: whether an x-axis column selector is needed
 * - needsY: whether a y-axis column selector is needed
 * - xType: column type filter for x-axis ('numerical' or 'categorical')
 * - yType: column type filter for y-axis ('numerical' or 'categorical')
 */
function getAxisConfig(chartType) {
  switch (chartType) {
    case 'histogram':
      return { needsX: true, needsY: false, xType: 'numerical', yType: null }
    case 'scatter':
      return { needsX: true, needsY: true, xType: 'numerical', yType: 'numerical' }
    case 'line':
      return { needsX: true, needsY: true, xType: 'numerical', yType: 'numerical' }
    case 'bar':
      return { needsX: true, needsY: true, xType: 'categorical', yType: 'numerical' }
    case 'pie':
      return { needsX: true, needsY: false, xType: 'categorical', yType: null }
    case 'box':
      return { needsX: true, needsY: false, xType: 'numerical', yType: null }
    case 'heatmap':
      return { needsX: false, needsY: false, xType: null, yType: null }
    default:
      return { needsX: false, needsY: false, xType: null, yType: null }
  }
}

/**
 * Classifies columns into numerical and categorical based on dtype strings.
 * Numerical: int64, float64, int32, float32, etc.
 * Categorical: object, category, bool, string, etc.
 */
function classifyColumns(columns, dtypes) {
  const numerical = []
  const categorical = []

  for (const col of columns) {
    const dtype = (dtypes[col] || '').toLowerCase()
    if (dtype.includes('int') || dtype.includes('float') || dtype.includes('numeric')) {
      numerical.push(col)
    } else {
      categorical.push(col)
    }
  }

  return { numerical, categorical }
}

function VisualizationPage() {
  const [chartType, setChartType] = useState('histogram')
  const [xColumn, setXColumn] = useState('')
  const [yColumn, setYColumn] = useState('')
  const [chartData, setChartData] = useState(null)
  const [columns, setColumns] = useState([])
  const [dtypes, setDtypes] = useState({})
  const [columnTypes, setColumnTypes] = useState({ numerical: [], categorical: [] })
  const [metadataLoaded, setMetadataLoaded] = useState(false)

  // Validation errors state: { fieldName: "error message" }
  const [validationErrors, setValidationErrors] = useState({})

  const { showLoading, startLoading, stopLoading } = useLoading()
  const { error, setError, clearError } = useError()

  const { filename } = useDatasetContext()

  // Fetch column metadata on mount
  useEffect(() => {
    if (!filename) return

    async function fetchMetadata() {
      try {
        const data = await api.get('/summary', { params: { filename } })
        const cols = data.columns || []
        const dt = data.dtypes || {}
        setColumns(cols)
        setDtypes(dt)
        setColumnTypes(classifyColumns(cols, dt))
        setMetadataLoaded(true)
      } catch (err) {
        const message = err?.message || 'Failed to load dataset metadata.'
        setError(message)
      }
    }

    fetchMetadata()
  }, [filename]) // eslint-disable-line react-hooks/exhaustive-deps

  // Get filtered columns for the current chart type
  const axisConfig = getAxisConfig(chartType)

  const xOptions = axisConfig.xType === 'numerical'
    ? columnTypes.numerical
    : axisConfig.xType === 'categorical'
      ? columnTypes.categorical
      : []

  const yOptions = axisConfig.yType === 'numerical'
    ? columnTypes.numerical
    : axisConfig.yType === 'categorical'
      ? columnTypes.categorical
      : []

  // Determine if applicable columns exist for the chart type
  const noApplicableColumns =
    axisConfig.needsX && xOptions.length === 0

  // Reset axis selections when chart type changes
  useEffect(() => {
    setXColumn('')
    setYColumn('')
    setChartData(null)
    setValidationErrors({})
  }, [chartType])

  // Fetch chart data when selections change
  const fetchChartData = useCallback(async () => {
    if (!filename) return
    if (chartType === 'heatmap') {
      // Heatmap needs no axis selections
      setValidationErrors({})
      clearError()
      startLoading()
      try {
        const result = await getVisualization(filename, 'heatmap')
        setChartData(result)
      } catch (err) {
        const message = err?.message || 'Failed to generate visualization.'
        setError(message)
        setChartData(null)
      } finally {
        stopLoading()
      }
      return
    }

    // Validate required column selections
    const errors = {}
    if (axisConfig.needsX && !xColumn) {
      errors.xColumn = `Please select a ${chartType === 'pie' ? 'column' : 'X axis column'}.`
    }
    if (axisConfig.needsY && !yColumn) {
      errors.yColumn = 'Please select a Y axis column.'
    }

    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors)
      setChartData(null)
      return
    }

    setValidationErrors({})
    clearError()
    startLoading()
    try {
      const result = await getVisualization(
        filename,
        chartType,
        xColumn || undefined,
        yColumn || undefined
      )
      setChartData(result)
    } catch (err) {
      const message = err?.message || 'Failed to generate visualization.'
      setError(message)
      setChartData(null)
    } finally {
      stopLoading()
    }
  }, [filename, chartType, xColumn, yColumn, axisConfig.needsX, axisConfig.needsY]) // eslint-disable-line react-hooks/exhaustive-deps

  // Trigger chart data fetch on selection change
  useEffect(() => {
    fetchChartData()
  }, [fetchChartData])

  if (!filename) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-800">Visualization</h1>
        <p className="mt-4 text-gray-600">
          No dataset loaded. Please upload a CSV file first.
        </p>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-800">Visualization</h1>
        <p className="mt-2 text-gray-600">
          Generate charts and explore your data visually.
        </p>
      </div>

      <ErrorAlert error={error} onDismiss={clearError} operation="Visualization" />

      {/* Chart type selector */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Chart type dropdown */}
          <div>
            <label htmlFor="chart-type" className="block text-sm font-medium text-gray-700 mb-1">
              Chart Type
            </label>
            <select
              id="chart-type"
              value={chartType}
              onChange={(e) => setChartType(e.target.value)}
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {CHART_TYPES.map((ct) => (
                <option key={ct.value} value={ct.value}>
                  {ct.label}
                </option>
              ))}
            </select>
          </div>

          {/* X-axis column selector */}
          {axisConfig.needsX && (
            <div>
              <label htmlFor="x-column" className="block text-sm font-medium text-gray-700 mb-1">
                {chartType === 'pie' ? 'Column' : 'X Axis'}
                <span className="ml-1 text-xs text-gray-400">
                  ({axisConfig.xType})
                </span>
              </label>
              <select
                id="x-column"
                value={xColumn}
                onChange={(e) => {
                  setXColumn(e.target.value)
                  setValidationErrors((prev) => { const { xColumn: _, ...rest } = prev; return rest })
                }}
                disabled={xOptions.length === 0}
                className={`w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-400 ${
                  validationErrors.xColumn ? 'border-red-500 ring-1 ring-red-500' : 'border-gray-300'
                }`}
                aria-invalid={!!validationErrors.xColumn}
                aria-describedby={validationErrors.xColumn ? 'x-column-error' : undefined}
              >
                <option value="">Select column...</option>
                {xOptions.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              {validationErrors.xColumn && (
                <p id="x-column-error" className="mt-1 text-xs text-red-600 flex items-center gap-1">
                  <svg className="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  {validationErrors.xColumn}
                </p>
              )}
            </div>
          )}

          {/* Y-axis column selector */}
          {axisConfig.needsY && (
            <div>
              <label htmlFor="y-column" className="block text-sm font-medium text-gray-700 mb-1">
                Y Axis
                <span className="ml-1 text-xs text-gray-400">
                  ({axisConfig.yType})
                </span>
              </label>
              <select
                id="y-column"
                value={yColumn}
                onChange={(e) => {
                  setYColumn(e.target.value)
                  setValidationErrors((prev) => { const { yColumn: _, ...rest } = prev; return rest })
                }}
                disabled={yOptions.length === 0}
                className={`w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-400 ${
                  validationErrors.yColumn ? 'border-red-500 ring-1 ring-red-500' : 'border-gray-300'
                }`}
                aria-invalid={!!validationErrors.yColumn}
                aria-describedby={validationErrors.yColumn ? 'y-column-error' : undefined}
              >
                <option value="">Select column...</option>
                {yOptions.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              {validationErrors.yColumn && (
                <p id="y-column-error" className="mt-1 text-xs text-red-600 flex items-center gap-1">
                  <svg className="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  {validationErrors.yColumn}
                </p>
              )}
            </div>
          )}
        </div>

        {/* No applicable columns message */}
        {metadataLoaded && noApplicableColumns && (
          <div className="rounded-md bg-yellow-50 border border-yellow-200 p-3 text-sm text-yellow-800">
            <p>
              No applicable {axisConfig.xType} columns exist in the dataset for the selected chart type.
              {axisConfig.xType === 'numerical'
                ? ' Try a chart type that uses categorical data (e.g., Pie, Bar).'
                : ' Try a chart type that uses numerical data (e.g., Histogram, Scatter).'}
            </p>
          </div>
        )}
      </div>

      {/* Loading spinner */}
      <LoadingSpinner loading={showLoading} message="Generating visualization..." />

      {/* Chart display */}
      {chartData && !showLoading && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <ChartRenderer
            type={chartData.chart_type}
            data={chartData.data}
            xKey={xColumn || 'x'}
            yKey={yColumn || 'y'}
            title={`${CHART_TYPES.find((ct) => ct.value === chartType)?.label || chartType}${xColumn ? ` — ${xColumn}` : ''}${yColumn ? ` vs ${yColumn}` : ''}`}
          />
        </div>
      )}
    </div>
  )
}

export default VisualizationPage
