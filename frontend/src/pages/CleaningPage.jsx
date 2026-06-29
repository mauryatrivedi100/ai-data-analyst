import { useState, useEffect, useCallback } from 'react'
import api from '../services/api'
import { useDatasetContext } from '../contexts/DatasetContext'
import {
  removeMissingRows,
  fillMean,
  fillMedian,
  fillMode,
  removeDuplicates,
  detectOutliers,
  removeOutliers,
  dropColumns,
  renameColumn,
  convertType,
  labelEncode,
  oneHotEncode,
  standardScale,
  minMaxScale,
  downloadCleaned,
} from '../services/cleaningService'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorAlert from '../components/ErrorAlert'

function CleaningPage() {
  // Dataset context for global filename
  const { filename: contextFilename } = useDatasetContext()

  // Dataset metadata
  const [filename, setFilename] = useState('')
  const [columns, setColumns] = useState([])
  const [dtypes, setDtypes] = useState({})
  const [missingValues, setMissingValues] = useState({})

  // UI state
  const [loading, setLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [error, setError] = useState(null)
  const [errorOperation, setErrorOperation] = useState('')
  const [successMessage, setSuccessMessage] = useState('')

  // Missing values form state
  const [missingColumn, setMissingColumn] = useState('')

  // Outlier form state
  const [outlierColumn, setOutlierColumn] = useState('')
  const [outlierInfo, setOutlierInfo] = useState(null)

  // Column operations form state
  const [dropColumnsSelected, setDropColumnsSelected] = useState([])
  const [renameOldName, setRenameOldName] = useState('')
  const [renameNewName, setRenameNewName] = useState('')
  const [convertColumn, setConvertColumn] = useState('')
  const [convertTargetType, setConvertTargetType] = useState('int')

  // Validation errors state: { fieldName: "error message" }
  const [validationErrors, setValidationErrors] = useState({})

  // Encoding/Scaling form state
  const [encodeColumn, setEncodeColumn] = useState('')

  // Derive numerical and categorical columns from dtypes
  const numericalColumns = columns.filter((col) => {
    const dtype = dtypes[col] || ''
    return dtype.includes('int') || dtype.includes('float')
  })
  const categoricalColumns = columns.filter((col) => {
    const dtype = dtypes[col] || ''
    return !dtype.includes('int') && !dtype.includes('float')
  })

  // Fetch dataset metadata on mount
  const fetchMetadata = useCallback(async (file) => {
    try {
      const data = await api.get('/summary', { params: { filename: file } })
      setColumns(data.columns || [])
      setDtypes(data.dtypes || {})
      setMissingValues(data.missing_values || {})
    } catch (err) {
      setError(err.message || 'Failed to load dataset metadata')
      setErrorOperation('Loading metadata')
    }
  }, [])

  useEffect(() => {
    if (contextFilename) {
      setFilename(contextFilename)
      fetchMetadata(contextFilename)
    }
  }, [contextFilename, fetchMetadata])

  // Helper: execute a cleaning operation with loading/error handling
  const executeOperation = async (operationFn, operationName, confirmMessage) => {
    if (confirmMessage && !window.confirm(confirmMessage)) {
      return null
    }

    setLoading(true)
    setLoadingMessage(`Performing: ${operationName}...`)
    setError(null)
    setSuccessMessage('')

    try {
      const result = await operationFn()
      setSuccessMessage(
        result?.summary || `${operationName} completed successfully.`
      )
      // Refresh metadata after operation
      await fetchMetadata(filename)
      return result
    } catch (err) {
      setError(err.message || `${operationName} failed`)
      setErrorOperation(operationName)
      return null
    } finally {
      setLoading(false)
      setLoadingMessage('')
    }
  }

  // --- Missing Values Handlers ---
  const handleRemoveMissingRows = () => {
    executeOperation(
      () => removeMissingRows(filename),
      'Remove rows with missing values',
      'This will remove all rows containing missing values. Continue?'
    )
  }

  const handleFillMean = () => {
    if (!missingColumn) {
      setError('Please select a column first')
      setErrorOperation('Fill with mean')
      return
    }
    executeOperation(
      () => fillMean(filename, missingColumn),
      `Fill missing values in "${missingColumn}" with mean`
    )
  }

  const handleFillMedian = () => {
    if (!missingColumn) {
      setError('Please select a column first')
      setErrorOperation('Fill with median')
      return
    }
    executeOperation(
      () => fillMedian(filename, missingColumn),
      `Fill missing values in "${missingColumn}" with median`
    )
  }

  const handleFillMode = () => {
    if (!missingColumn) {
      setError('Please select a column first')
      setErrorOperation('Fill with mode')
      return
    }
    executeOperation(
      () => fillMode(filename, missingColumn),
      `Fill missing values in "${missingColumn}" with mode`
    )
  }

  // --- Duplicate Handlers ---
  const handleRemoveDuplicates = () => {
    executeOperation(
      () => removeDuplicates(filename),
      'Remove duplicate rows'
    )
  }

  // --- Outlier Handlers ---
  const handleDetectOutliers = async () => {
    if (!outlierColumn) {
      setError('Please select a numerical column first')
      setErrorOperation('Detect outliers')
      return
    }
    setOutlierInfo(null)
    const result = await executeOperation(
      () => detectOutliers(filename, outlierColumn),
      `Detect outliers in "${outlierColumn}"`
    )
    if (result) {
      setOutlierInfo(result)
    }
  }

  const handleRemoveOutliers = () => {
    if (!outlierColumn) {
      setError('Please select a numerical column first')
      setErrorOperation('Remove outliers')
      return
    }
    executeOperation(
      () => removeOutliers(filename, outlierColumn),
      `Remove outliers from "${outlierColumn}"`,
      `This will remove rows with outlier values in "${outlierColumn}". Continue?`
    ).then(() => setOutlierInfo(null))
  }

  // --- Column Operations Handlers ---
  const handleDropColumns = () => {
    if (dropColumnsSelected.length === 0) {
      setError('Please select at least one column to drop')
      setErrorOperation('Drop columns')
      return
    }
    if (dropColumnsSelected.length >= columns.length) {
      setError('Cannot drop all columns. At least one column must remain.')
      setErrorOperation('Drop columns')
      return
    }
    executeOperation(
      () => dropColumns(filename, dropColumnsSelected),
      `Drop columns: ${dropColumnsSelected.join(', ')}`,
      `This will permanently remove ${dropColumnsSelected.length} column(s). Continue?`
    ).then((result) => {
      if (result) setDropColumnsSelected([])
    })
  }

  const handleRenameColumn = () => {
    const errors = {}

    if (!renameOldName) {
      errors.renameOldName = 'Please select a column to rename.'
    }
    if (!renameNewName.trim()) {
      errors.renameNewName = 'New column name cannot be empty.'
    } else if (renameNewName.trim().length > 128) {
      errors.renameNewName = 'Column name must be at most 128 characters.'
    } else if (columns.includes(renameNewName.trim()) && renameNewName.trim() !== renameOldName) {
      errors.renameNewName = 'A column with that name already exists.'
    }

    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors)
      return
    }

    setValidationErrors({})
    executeOperation(
      () => renameColumn(filename, renameOldName, renameNewName.trim()),
      `Rename "${renameOldName}" to "${renameNewName.trim()}"`
    ).then((result) => {
      if (result) {
        setRenameOldName('')
        setRenameNewName('')
      }
    })
  }

  const handleConvertType = () => {
    if (!convertColumn) {
      setError('Please select a column to convert')
      setErrorOperation('Convert type')
      return
    }
    executeOperation(
      () => convertType(filename, convertColumn, convertTargetType),
      `Convert "${convertColumn}" to ${convertTargetType}`
    )
  }

  // --- Encoding/Scaling Handlers ---
  const handleLabelEncode = () => {
    if (!encodeColumn) {
      setError('Please select a column')
      setErrorOperation('Label encode')
      return
    }
    executeOperation(
      () => labelEncode(filename, encodeColumn),
      `Label encode "${encodeColumn}"`
    )
  }

  const handleOneHotEncode = () => {
    if (!encodeColumn) {
      setError('Please select a column')
      setErrorOperation('One-hot encode')
      return
    }
    executeOperation(
      () => oneHotEncode(filename, encodeColumn),
      `One-hot encode "${encodeColumn}"`
    )
  }

  const handleStandardScale = () => {
    if (!encodeColumn) {
      setError('Please select a column')
      setErrorOperation('Standard scale')
      return
    }
    executeOperation(
      () => standardScale(filename, encodeColumn),
      `Standard scale "${encodeColumn}"`
    )
  }

  const handleMinMaxScale = () => {
    if (!encodeColumn) {
      setError('Please select a column')
      setErrorOperation('Min-max scale')
      return
    }
    executeOperation(
      () => minMaxScale(filename, encodeColumn),
      `Min-max scale "${encodeColumn}"`
    )
  }

  // --- Download Handler ---
  const handleDownload = async () => {
    setLoading(true)
    setLoadingMessage('Preparing download...')
    setError(null)
    setSuccessMessage('')
    try {
      await downloadCleaned(filename)
      setSuccessMessage('Download started successfully.')
    } catch (err) {
      setError(err.message || 'Download failed')
      setErrorOperation('Download')
    } finally {
      setLoading(false)
      setLoadingMessage('')
    }
  }

  // Multi-select toggle for drop columns
  const toggleDropColumn = (col) => {
    setDropColumnsSelected((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    )
  }

  if (!filename) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-800">Data Cleaning</h1>
        <p className="mt-4 text-gray-600">
          No dataset loaded. Please upload a dataset first.
        </p>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-800">Data Cleaning</h1>
        <p className="mt-1 text-gray-600">
          Clean and transform your dataset: <span className="font-medium">{filename}</span>
        </p>
      </div>

      {/* Loading Spinner */}
      <LoadingSpinner loading={loading} message={loadingMessage} />

      {/* Error Alert */}
      <ErrorAlert
        error={error}
        onDismiss={() => setError(null)}
        operation={errorOperation}
      />

      {/* Success Message */}
      {successMessage && (
        <div
          className="rounded-md border border-green-300 bg-green-50 p-4 text-sm text-green-800"
          role="status"
        >
          <p className="font-medium">Operation Successful</p>
          <p>{successMessage}</p>
        </div>
      )}

      {/* Section 1: Missing Values */}
      <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Missing Value Handling
        </h2>

        <div className="space-y-4">
          <div>
            <label htmlFor="missing-column" className="block text-sm font-medium text-gray-700 mb-1">
              Select Column
            </label>
            <select
              id="missing-column"
              value={missingColumn}
              onChange={(e) => setMissingColumn(e.target.value)}
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">-- Select a column --</option>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col} {missingValues[col] > 0 ? `(${missingValues[col]} missing)` : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={handleRemoveMissingRows}
              disabled={loading}
              className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Remove Rows with Missing Values
            </button>
            <button
              onClick={handleFillMean}
              disabled={loading || !missingColumn}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Fill with Mean
            </button>
            <button
              onClick={handleFillMedian}
              disabled={loading || !missingColumn}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Fill with Median
            </button>
            <button
              onClick={handleFillMode}
              disabled={loading || !missingColumn}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Fill with Mode
            </button>
          </div>
        </div>
      </section>

      {/* Section 2: Duplicates */}
      <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Duplicate Removal
        </h2>
        <button
          onClick={handleRemoveDuplicates}
          disabled={loading}
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Remove Duplicate Rows
        </button>
      </section>

      {/* Section 3: Outliers */}
      <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Outlier Detection &amp; Removal
        </h2>

        <div className="space-y-4">
          <div>
            <label htmlFor="outlier-column" className="block text-sm font-medium text-gray-700 mb-1">
              Select Numerical Column
            </label>
            <select
              id="outlier-column"
              value={outlierColumn}
              onChange={(e) => {
                setOutlierColumn(e.target.value)
                setOutlierInfo(null)
              }}
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">-- Select a numerical column --</option>
              {numericalColumns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={handleDetectOutliers}
              disabled={loading || !outlierColumn}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Detect Outliers
            </button>
            <button
              onClick={handleRemoveOutliers}
              disabled={loading || !outlierColumn}
              className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Remove Outliers
            </button>
          </div>

          {outlierInfo && outlierInfo.summary && (
            <div className="mt-3 rounded-md bg-blue-50 border border-blue-200 p-3 text-sm text-blue-800">
              <p className="font-medium">Outlier Detection Results:</p>
              <p>{outlierInfo.summary}</p>
            </div>
          )}
        </div>
      </section>

      {/* Section 4: Column Operations */}
      <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Column Operations
        </h2>

        <div className="space-y-6">
          {/* Drop Columns */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Drop Columns</h3>
            <div className="flex flex-wrap gap-2 mb-3 max-h-32 overflow-y-auto border border-gray-200 rounded-md p-2">
              {columns.map((col) => (
                <label
                  key={col}
                  className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium cursor-pointer border ${
                    dropColumnsSelected.includes(col)
                      ? 'bg-red-100 border-red-300 text-red-800'
                      : 'bg-gray-100 border-gray-200 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={dropColumnsSelected.includes(col)}
                    onChange={() => toggleDropColumn(col)}
                  />
                  {col}
                </label>
              ))}
            </div>
            {dropColumnsSelected.length > 0 && (
              <p className="text-xs text-gray-500 mb-2">
                Selected: {dropColumnsSelected.join(', ')}
              </p>
            )}
            <button
              onClick={handleDropColumns}
              disabled={loading || dropColumnsSelected.length === 0}
              className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Drop Selected Columns
            </button>
          </div>

          {/* Rename Column */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Rename Column</h3>
            <div className="flex flex-wrap items-end gap-3">
              <div>
                <label htmlFor="rename-old" className="block text-xs text-gray-600 mb-1">
                  Column
                </label>
                <select
                  id="rename-old"
                  value={renameOldName}
                  onChange={(e) => {
                    setRenameOldName(e.target.value)
                    setValidationErrors((prev) => { const { renameOldName: _, ...rest } = prev; return rest })
                  }}
                  className={`rounded-md border px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 ${
                    validationErrors.renameOldName ? 'border-red-500 ring-1 ring-red-500' : 'border-gray-300'
                  }`}
                  aria-invalid={!!validationErrors.renameOldName}
                  aria-describedby={validationErrors.renameOldName ? 'rename-old-error' : undefined}
                >
                  <option value="">-- Select --</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                {validationErrors.renameOldName && (
                  <p id="rename-old-error" className="mt-1 text-xs text-red-600 flex items-center gap-1">
                    <svg className="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    {validationErrors.renameOldName}
                  </p>
                )}
              </div>
              <div>
                <label htmlFor="rename-new" className="block text-xs text-gray-600 mb-1">
                  New Name
                </label>
                <input
                  id="rename-new"
                  type="text"
                  value={renameNewName}
                  onChange={(e) => {
                    setRenameNewName(e.target.value)
                    setValidationErrors((prev) => { const { renameNewName: _, ...rest } = prev; return rest })
                  }}
                  placeholder="New column name"
                  maxLength={128}
                  className={`rounded-md border px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 ${
                    validationErrors.renameNewName ? 'border-red-500 ring-1 ring-red-500' : 'border-gray-300'
                  }`}
                  aria-invalid={!!validationErrors.renameNewName}
                  aria-describedby={validationErrors.renameNewName ? 'rename-new-error' : undefined}
                />
                {validationErrors.renameNewName && (
                  <p id="rename-new-error" className="mt-1 text-xs text-red-600 flex items-center gap-1">
                    <svg className="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    {validationErrors.renameNewName}
                  </p>
                )}
              </div>
              <button
                onClick={handleRenameColumn}
                disabled={loading}
                className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Rename
              </button>
            </div>
          </div>

          {/* Convert Type */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Convert Column Type</h3>
            <div className="flex flex-wrap items-end gap-3">
              <div>
                <label htmlFor="convert-column" className="block text-xs text-gray-600 mb-1">
                  Column
                </label>
                <select
                  id="convert-column"
                  value={convertColumn}
                  onChange={(e) => setConvertColumn(e.target.value)}
                  className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="">-- Select --</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="convert-type" className="block text-xs text-gray-600 mb-1">
                  Target Type
                </label>
                <select
                  id="convert-type"
                  value={convertTargetType}
                  onChange={(e) => setConvertTargetType(e.target.value)}
                  className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="int">Integer</option>
                  <option value="float">Float</option>
                  <option value="string">String</option>
                  <option value="datetime">Datetime</option>
                </select>
              </div>
              <button
                onClick={handleConvertType}
                disabled={loading || !convertColumn}
                className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Convert
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Section 5: Encoding and Scaling */}
      <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Encoding &amp; Scaling
        </h2>

        <div className="space-y-4">
          <div>
            <label htmlFor="encode-column" className="block text-sm font-medium text-gray-700 mb-1">
              Select Column
            </label>
            <select
              id="encode-column"
              value={encodeColumn}
              onChange={(e) => setEncodeColumn(e.target.value)}
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">-- Select a column --</option>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col} ({dtypes[col]})
                </option>
              ))}
            </select>
          </div>

          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">Encoding (Categorical)</p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleLabelEncode}
                disabled={loading || !encodeColumn}
                className="rounded-md bg-purple-600 px-4 py-2 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Label Encode
              </button>
              <button
                onClick={handleOneHotEncode}
                disabled={loading || !encodeColumn}
                className="rounded-md bg-purple-600 px-4 py-2 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                One-Hot Encode
              </button>
            </div>
          </div>

          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">Scaling (Numerical)</p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleStandardScale}
                disabled={loading || !encodeColumn}
                className="rounded-md bg-teal-600 px-4 py-2 text-sm font-medium text-white hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Standard Scale
              </button>
              <button
                onClick={handleMinMaxScale}
                disabled={loading || !encodeColumn}
                className="rounded-md bg-teal-600 px-4 py-2 text-sm font-medium text-white hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Min-Max Scale
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Section 6: Download Cleaned Dataset */}
      <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Download Cleaned Dataset
        </h2>
        <button
          onClick={handleDownload}
          disabled={loading}
          className="rounded-md bg-green-600 px-6 py-3 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Download as CSV
        </button>
      </section>
    </div>
  )
}

export default CleaningPage
