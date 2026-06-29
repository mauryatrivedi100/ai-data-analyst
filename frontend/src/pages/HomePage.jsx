import { useState, useCallback } from 'react'
import FileUploader from '../components/FileUploader'
import DataTable from '../components/DataTable'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'
import { useDataset } from '../hooks/useDataset'
import { useDatasetContext } from '../contexts/DatasetContext'
import { useLoading } from '../hooks/useLoading'
import { useError } from '../hooks/useError'
import { uploadFile } from '../services/uploadService'
import api from '../services/api'

function HomePage() {
  const { dataset, setDataset, clearDataset } = useDataset()
  const { filename, setFilename, setDatasetMetadata, clearDataset: clearGlobalDataset, hasDataset } = useDatasetContext()
  const { showLoading, startLoading, stopLoading } = useLoading()
  const { error, setError, clearError } = useError()
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploading, setUploading] = useState(false)

  const handleUpload = useCallback(async (file) => {
    clearError()
    setUploadProgress(0)
    setUploading(true)
    startLoading()

    try {
      const response = await uploadFile(file, (progress) => {
        setUploadProgress(progress)
      })
      setDataset(response)
      // Update global dataset context (syncs with localStorage)
      if (response.filename) {
        setFilename(response.filename)
        setDatasetMetadata({
          originalName: response.originalName || response.original_name,
          rows: response.rows,
          columns: response.columns,
          columnNames: response.columnNames || response.column_names,
          fileSize: response.fileSize || response.file_size,
        })
      }
    } catch (err) {
      const message = err?.message || 'An unexpected error occurred during upload.'
      setError(message)
    } finally {
      setUploading(false)
      stopLoading()
    }
  }, [clearError, startLoading, stopLoading, setDataset, setError, setFilename, setDatasetMetadata])

  /**
   * Clear everything — local state, global context, localStorage, and server session.
   * This allows uploading a fresh file and starting a new analysis.
   */
  const handleStartNewAnalysis = useCallback(async () => {
    const confirmed = window.confirm(
      'This will remove the current dataset and all analysis results. Are you sure?'
    )
    if (!confirmed) return

    // Clear server-side session data for the current file
    if (filename) {
      try {
        await api.post('/reset-session', { filename })
      } catch {
        // If server reset fails, still clear client state
      }
    }

    // Clear all client state
    clearDataset()
    clearGlobalDataset()
    clearError()
  }, [filename, clearDataset, clearGlobalDataset, clearError])

  // Determine if a dataset is loaded (either from this session or from localStorage on page load)
  const datasetLoaded = dataset || hasDataset

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      {/* Landing section */}
      <section className="text-center space-y-3">
        <h1 className="text-3xl font-bold text-gray-800">
          AI Data Analyst
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload a CSV dataset to explore, clean, visualize, and model your data — all without writing code.
          Get AI-powered insights and generate PDF reports in minutes.
        </p>
      </section>

      {/* File uploader — only shown when NO dataset is loaded */}
      {!datasetLoaded && (
        <section>
          <FileUploader
            onUpload={handleUpload}
            uploading={uploading}
            progress={uploadProgress}
            error={null}
            onClearError={clearError}
          />
        </section>
      )}

      {/* Loading spinner */}
      <LoadingSpinner loading={showLoading} message="Uploading and processing your file..." />

      {/* Error alert */}
      {error && !uploading && (
        <ErrorAlert
          error={error}
          onDismiss={clearError}
          operation="File upload"
        />
      )}

      {/* Dataset preview on successful upload */}
      {dataset && !uploading && (
        <section className="space-y-4">
          {/* File info summary */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-green-800 mb-2">
              Dataset Loaded
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
              <div>
                <span className="text-gray-500 block">File Name</span>
                <span className="font-medium text-gray-800">{dataset.originalName || dataset.filename}</span>
              </div>
              <div>
                <span className="text-gray-500 block">File Size</span>
                <span className="font-medium text-gray-800">{dataset.fileSize}</span>
              </div>
              <div>
                <span className="text-gray-500 block">Rows</span>
                <span className="font-medium text-gray-800">{dataset.rows.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-gray-500 block">Columns</span>
                <span className="font-medium text-gray-800">{dataset.columns}</span>
              </div>
            </div>
          </div>

          {/* Data table preview */}
          <div>
            <h3 className="text-md font-semibold text-gray-700 mb-2">
              Data Preview (first 20 rows)
            </h3>
            <DataTable
              data={dataset.preview}
              columns={dataset.columnNames}
              maxRows={20}
            />
          </div>
        </section>
      )}

      {/* When dataset is loaded but no preview (e.g. page reload), show basic info */}
      {!dataset && hasDataset && !uploading && (
        <section>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-blue-800 mb-1">
              Dataset Active
            </h2>
            <p className="text-sm text-blue-700">
              Current dataset: <span className="font-medium">{filename}</span>
            </p>
            <p className="text-sm text-blue-600 mt-1">
              Navigate to Analysis, Cleaning, Visualization, ML, or Insights pages to work with your data.
            </p>
          </div>
        </section>
      )}

      {/* Start New Analysis button — shown when a dataset is loaded */}
      {datasetLoaded && !uploading && (
        <section className="pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">
              Want to analyze a different dataset?
            </p>
            <button
              onClick={handleStartNewAnalysis}
              className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 transition-colors"
            >
              Start New Analysis
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            This will remove the current dataset and clear all analysis results (cleaning history, model metrics, insights).
          </p>
        </section>
      )}
    </div>
  )
}

export default HomePage
