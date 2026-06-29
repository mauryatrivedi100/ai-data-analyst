import { useState } from 'react'
import { useDatasetContext } from '../contexts/DatasetContext'
import { downloadReport } from '../services/reportService'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'

/**
 * ReportPage — Generate and download a PDF report of the analysis.
 *
 * Provides a "Generate Report" button that calls the backend to compile
 * a PDF from all completed analysis steps, then triggers a browser download.
 * (Requirements 14.1, 14.4, 14.5)
 */
function ReportPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [downloadedFile, setDownloadedFile] = useState(null)

  const { filename } = useDatasetContext()

  async function handleGenerateReport() {
    setError(null)
    setDownloadedFile(null)
    setLoading(true)
    try {
      const name = await downloadReport(filename)
      setDownloadedFile(name)
    } catch (err) {
      let message = 'Failed to generate report. Please try again.'
      if (err?.response?.data) {
        // The error response is a blob, need to parse it
        try {
          const text = await err.response.data.text()
          const parsed = JSON.parse(text)
          message = parsed.error || message
        } catch {
          // If blob parsing fails, use default message
        }
      } else if (err?.message) {
        message = err.message
      }
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  // No dataset uploaded — prompt user to upload first
  if (!filename) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-800">Report</h1>
        <p className="mt-2 text-gray-600">
          Generate and download a PDF report of your analysis.
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
      <h1 className="text-3xl font-bold text-gray-800">Report</h1>
      <p className="mt-2 text-gray-600">
        Generate and download a PDF report of your analysis.
      </p>

      {/* Dataset info */}
      <div className="mt-4 text-sm text-gray-500">
        Dataset: <span className="font-medium text-gray-700">{filename}</span>
      </div>

      {/* Generate button */}
      <div className="mt-6">
        <button
          type="button"
          onClick={handleGenerateReport}
          disabled={loading}
          className="rounded-md bg-blue-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Generating…' : 'Generate Report'}
        </button>
      </div>

      {/* Loading indicator */}
      <LoadingSpinner loading={loading} message="Generating PDF report… This may take a few seconds." />

      {/* Error display */}
      {error && (
        <div className="mt-4">
          <ErrorAlert
            error={error}
            onDismiss={() => setError(null)}
            operation="Generate Report"
          />
        </div>
      )}

      {/* Success message */}
      {downloadedFile && !loading && (
        <div className="mt-4 rounded-md border border-green-300 bg-green-50 p-4">
          <div className="flex items-center gap-2">
            <svg className="h-5 w-5 text-green-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
            </svg>
            <p className="text-sm font-medium text-green-800">
              Report downloaded successfully: {downloadedFile}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ReportPage
