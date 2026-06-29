import { useState } from 'react'
import { useDatasetContext } from '../contexts/DatasetContext'
import { generateInsights } from '../services/insightsService'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'

/**
 * InsightsPage — AI-powered insights generation and display.
 *
 * Allows users to generate natural language insights about their dataset
 * using the Gemini API. Displays results in labeled sections:
 * overview, key observations, business insights, potential risks, recommendations.
 * (Requirements 13.1–13.5)
 */
function InsightsPage() {
  const [insights, setInsights] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const { filename } = useDatasetContext()

  async function handleGenerateInsights() {
    setError(null)
    setLoading(true)
    try {
      const data = await generateInsights(filename)
      setInsights(data)
    } catch (err) {
      const message =
        typeof err === 'string'
          ? err
          : err?.message || 'Failed to generate insights. Please try again.'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  // No dataset uploaded — prompt user to upload first
  if (!filename) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-800">AI Insights</h1>
        <p className="mt-2 text-gray-600">
          Generate AI-powered insights about your dataset.
        </p>
        <div className="mt-6 rounded-md border border-yellow-300 bg-yellow-50 p-4">
          <p className="text-sm text-yellow-800">
            No dataset uploaded. Please go to the{' '}
            <span className="font-medium">Home</span> page and upload a CSV file
            first.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-800">AI Insights</h1>
      <p className="mt-2 text-gray-600">
        Generate AI-powered insights about your dataset.
      </p>

      {/* Generate button */}
      <div className="mt-6">
        <button
          type="button"
          onClick={handleGenerateInsights}
          disabled={loading}
          className="rounded-md bg-blue-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Generating…' : 'Generate Insights'}
        </button>
      </div>

      {/* Loading indicator */}
      <LoadingSpinner loading={loading} message="Generating AI insights… This may take up to 30 seconds." />

      {/* Error display */}
      {error && (
        <div className="mt-4">
          <ErrorAlert
            error={error}
            onDismiss={() => setError(null)}
            operation="Generate Insights"
          />
        </div>
      )}

      {/* Insights sections */}
      {insights && !loading && (
        <div className="mt-6 space-y-6">
          {/* Overview */}
          <InsightSection
            title="Overview"
            content={insights.overview}
            icon={
              <svg className="h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
              </svg>
            }
          />

          {/* Key Observations */}
          <InsightSection
            title="Key Observations"
            content={insights.observations}
            icon={
              <svg className="h-5 w-5 text-green-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                <path fillRule="evenodd" d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41z" clipRule="evenodd" />
              </svg>
            }
          />

          {/* Business Insights */}
          <InsightSection
            title="Business Insights"
            content={insights.business_insights}
            icon={
              <svg className="h-5 w-5 text-purple-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M1 2.75A.75.75 0 011.75 2h16.5a.75.75 0 010 1.5H18v8.75A2.75 2.75 0 0115.25 15h-1.072l.798 3.06a.75.75 0 01-1.452.38L13.41 18H6.59l-.114.44a.75.75 0 01-1.452-.38L5.822 15H4.75A2.75 2.75 0 012 12.25V3.5h-.25A.75.75 0 011 2.75z" clipRule="evenodd" />
              </svg>
            }
          />

          {/* Potential Risks */}
          <InsightSection
            title="Potential Risks"
            content={insights.risks}
            icon={
              <svg className="h-5 w-5 text-amber-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
              </svg>
            }
          />

          {/* Recommendations */}
          <InsightSection
            title="Recommendations"
            content={insights.recommendations}
            icon={
              <svg className="h-5 w-5 text-teal-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M10 2a.75.75 0 01.75.75v.258a33.186 33.186 0 016.668.83.75.75 0 01-.336 1.461 31.28 31.28 0 00-1.103-.232l1.702 7.545a.75.75 0 01-.387.832A4.981 4.981 0 0115 14c-.825 0-1.606-.2-2.294-.556a.75.75 0 01-.387-.832l1.77-7.849a31.743 31.743 0 00-3.339-.254v11.505l5.036.013a.75.75 0 010 1.5l-11.572-.03a.75.75 0 01-.002-1.5L9.25 16.014V4.508a31.715 31.715 0 00-3.339.254l1.77 7.85a.75.75 0 01-.387.83A4.981 4.981 0 015 14c-.825 0-1.606-.2-2.294-.556a.75.75 0 01-.387-.832l1.702-7.545c-.372.06-.742.126-1.103.232a.75.75 0 01-.336-1.461 33.186 33.186 0 016.668-.83V2.75A.75.75 0 0110 2z" clipRule="evenodd" />
              </svg>
            }
          />
        </div>
      )}
    </div>
  )
}

/**
 * InsightSection — Renders a single insight category as a styled card.
 */
function InsightSection({ title, content, icon }) {
  if (!content) return null

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
      </div>
      <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
        {content}
      </div>
    </div>
  )
}

export default InsightsPage
