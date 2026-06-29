import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import api from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'
import MetricsCard from '../components/MetricsCard'
import { useDatasetContext } from '../contexts/DatasetContext'
import { useLoading } from '../hooks/useLoading'
import { useError } from '../hooks/useError'

function AnalysisDashboardPage() {
  const [summary, setSummary] = useState(null)
  const [statistics, setStatistics] = useState(null)
  const { showLoading, startLoading, stopLoading } = useLoading()
  const { error, setError, clearError } = useError()

  const { filename } = useDatasetContext()

  useEffect(() => {
    if (!filename) return

    async function fetchData() {
      clearError()
      startLoading()
      try {
        const [summaryData, statsData] = await Promise.all([
          api.get('/summary', { params: { filename } }),
          api.get('/statistics', { params: { filename } }),
        ])
        setSummary(summaryData)
        setStatistics(statsData)
      } catch (err) {
        const message = err?.message || 'Failed to load dataset analysis.'
        setError(message)
      } finally {
        stopLoading()
      }
    }

    fetchData()
  }, [filename]) // eslint-disable-line react-hooks/exhaustive-deps

  // No dataset uploaded yet
  if (!filename) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-gray-800">Analysis Dashboard</h1>
        <div className="mt-8 text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-gray-600 text-lg">No dataset uploaded yet.</p>
          <p className="text-gray-500 mt-2">
            Please{' '}
            <Link to="/" className="text-blue-600 hover:underline font-medium">
              upload a dataset
            </Link>{' '}
            to view the analysis dashboard.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <h1 className="text-3xl font-bold text-gray-800">Analysis Dashboard</h1>

      <LoadingSpinner loading={showLoading} message="Loading dataset analysis..." />

      {error && (
        <ErrorAlert error={error} onDismiss={clearError} operation="Dataset analysis" />
      )}

      {summary && (
        <>
          {/* Summary metrics grid */}
          <section className="space-y-4">
            <h2 className="text-xl font-semibold text-gray-700">Dataset Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              <MetricsCard label="Row Count" value={summary.row_count?.toLocaleString()} />
              <MetricsCard label="Column Count" value={summary.column_count} />
              <MetricsCard label="Duplicate Rows" value={summary.duplicate_rows?.toLocaleString()} />
              <MetricsCard label="Memory Usage" value={summary.memory_usage} />
            </div>
          </section>

          {/* Column info table */}
          <section className="space-y-4">
            <h2 className="text-xl font-semibold text-gray-700">Column Information</h2>
            <div className="overflow-x-auto border border-gray-200 rounded-lg">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Column Name</th>
                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Data Type</th>
                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Missing Values</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {summary.columns?.map((col, idx) => (
                    <tr
                      key={col}
                      className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                    >
                      <td className="px-4 py-2 text-gray-800 font-medium">{col}</td>
                      <td className="px-4 py-2 text-gray-600">
                        {summary.dtypes?.[col] || '—'}
                      </td>
                      <td className="px-4 py-2 text-gray-600">
                        {summary.missing_values?.[col] ?? 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {statistics && (
        <>
          {/* Numerical statistics */}
          {statistics.numerical && Object.keys(statistics.numerical).length > 0 && (
            <section className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-700">Numerical Statistics</h2>
              <div className="overflow-x-auto border border-gray-200 rounded-lg">
                <table className="min-w-full divide-y divide-gray-200 text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">Column</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Mean</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Median</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Std Dev</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Min</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Max</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Q1</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Q2</th>
                      <th className="px-4 py-3 text-right font-semibold text-gray-700">Q3</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {Object.entries(statistics.numerical).map(([col, stats], idx) => {
                      const allNull = stats.mean == null && stats.median == null && stats.std == null
                        && stats.min == null && stats.max == null
                      return (
                        <tr key={col} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          <td className="px-4 py-2 text-gray-800 font-medium">{col}</td>
                          {allNull ? (
                            <td colSpan={8} className="px-4 py-2 text-center text-gray-400 italic">
                              No data available
                            </td>
                          ) : (
                            <>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.mean)}</td>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.median)}</td>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.std)}</td>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.min)}</td>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.max)}</td>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.q1)}</td>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.q2)}</td>
                              <td className="px-4 py-2 text-right text-gray-600">{formatStat(stats.q3)}</td>
                            </>
                          )}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Categorical statistics */}
          {statistics.categorical && Object.keys(statistics.categorical).length > 0 && (
            <section className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-700">Categorical Statistics</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(statistics.categorical).map(([col, stats]) => {
                  const allNull = stats.unique_count == null && (!stats.top_5 || stats.top_5.length === 0)
                  return (
                    <div
                      key={col}
                      className="border border-gray-200 rounded-lg p-4 bg-white"
                    >
                      <h3 className="font-semibold text-gray-800 mb-2">{col}</h3>
                      {allNull ? (
                        <p className="text-gray-400 italic">No data available</p>
                      ) : (
                        <>
                          <p className="text-sm text-gray-600 mb-2">
                            Unique values: <span className="font-medium">{stats.unique_count}</span>
                          </p>
                          {stats.top_5 && stats.top_5.length > 0 && (
                            <div>
                              <p className="text-sm text-gray-500 mb-1">Top 5 values:</p>
                              <ul className="space-y-1">
                                {stats.top_5.map((item, i) => (
                                  <li
                                    key={i}
                                    className="flex justify-between text-sm bg-gray-50 px-3 py-1 rounded"
                                  >
                                    <span className="text-gray-700">{item.value}</span>
                                    <span className="text-gray-500 font-medium">{item.count}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )
                })}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}

/**
 * Format a statistic value for display.
 * Returns "—" if the value is null/undefined, otherwise the number itself.
 */
function formatStat(value) {
  if (value == null) return '—'
  return value
}

export default AnalysisDashboardPage
