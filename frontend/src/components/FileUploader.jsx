import { useState, useRef, useCallback } from 'react'

const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50 MB
const MIN_FILE_SIZE = 1 // 1 byte

/**
 * Validates a file for CSV extension and size constraints.
 * Returns an error message string or null if valid.
 */
function validateFile(file) {
  if (!file) {
    return 'No file selected.'
  }

  const name = file.name || ''
  if (!name.toLowerCase().endsWith('.csv')) {
    return 'Invalid file type. Please upload a .csv file.'
  }

  if (file.size < MIN_FILE_SIZE) {
    return 'File is empty. Please upload a non-empty CSV file.'
  }

  if (file.size > MAX_FILE_SIZE) {
    return `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum size is 50 MB.`
  }

  return null
}

/**
 * FileUploader — Drag-and-drop zone and file picker with upload progress and validation.
 *
 * Props:
 *  - onUpload(file): called when a valid file is selected
 *  - uploading: boolean indicating upload in progress
 *  - progress: number 0-100 showing upload progress
 *  - error: string or null, displayed as upload error from parent
 *  - onClearError: callback to clear the error state
 */
function FileUploader({ onUpload, uploading = false, progress = 0, error = null, onClearError }) {
  const [dragOver, setDragOver] = useState(false)
  const [localError, setLocalError] = useState(null)
  const fileInputRef = useRef(null)

  const displayError = localError || error

  const handleFile = useCallback((file) => {
    // Clear any previous errors
    setLocalError(null)
    if (onClearError) onClearError()

    const validationError = validateFile(file)
    if (validationError) {
      setLocalError(validationError)
      return
    }

    onUpload(file)
  }, [onUpload, onClearError])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)

    const files = e.dataTransfer?.files
    if (files && files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  const handleFileInputChange = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFile(file)
    }
    // Reset the input so the same file can be re-selected
    e.target.value = ''
  }, [handleFile])

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="w-full">
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragOver
            ? 'border-indigo-500 bg-indigo-50'
            : 'border-gray-300 bg-white hover:border-gray-400'
        } ${uploading ? 'pointer-events-none opacity-60' : ''}`}
      >
        {/* Upload icon */}
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"
          />
        </svg>

        <p className="mt-3 text-sm text-gray-600">
          <span className="font-medium">Drag and drop</span> your CSV file here, or
        </p>

        {/* File picker button */}
        <button
          type="button"
          onClick={handleButtonClick}
          disabled={uploading}
          className="mt-3 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Browse Files
        </button>

        <p className="mt-2 text-xs text-gray-500">
          CSV files only, up to 50 MB
        </p>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileInputChange}
          className="hidden"
          aria-label="Upload CSV file"
        />
      </div>

      {/* Progress bar */}
      {uploading && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-gray-700">Uploading...</span>
            <span className="text-sm font-medium text-gray-700">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
              role="progressbar"
              aria-valuenow={progress}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>
      )}

      {/* Error display */}
      {displayError && !uploading && (
        <div className="mt-4 flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-md">
          <svg
            className="h-5 w-5 text-red-500 shrink-0 mt-0.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <p className="text-sm text-red-700 flex-1">{displayError}</p>
          <button
            type="button"
            onClick={() => {
              setLocalError(null)
              if (onClearError) onClearError()
            }}
            className="text-red-500 hover:text-red-700"
            aria-label="Dismiss error"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
    </div>
  )
}

export { validateFile }
export default FileUploader
