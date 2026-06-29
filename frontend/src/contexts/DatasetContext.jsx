import { createContext, useContext, useState, useCallback } from 'react'

/**
 * DatasetContext — Global context for sharing dataset state across all pages.
 *
 * Stores the current dataset filename and metadata, syncing with localStorage
 * for persistence across page reloads. All pages consume this context to
 * determine whether a dataset has been uploaded and to get the filename
 * for service calls.
 *
 * Requirements: 13.5, 15.1
 */
const DatasetContext = createContext(null)

export function DatasetProvider({ children }) {
  const [filename, setFilenameState] = useState(
    () => localStorage.getItem('dataset_filename') || null
  )
  const [metadata, setMetadata] = useState(null)

  /**
   * Set dataset filename and persist to localStorage.
   */
  const setFilename = useCallback((name) => {
    setFilenameState(name)
    if (name) {
      localStorage.setItem('dataset_filename', name)
    } else {
      localStorage.removeItem('dataset_filename')
    }
  }, [])

  /**
   * Update dataset metadata (rows, columns, originalName, etc.)
   */
  const setDatasetMetadata = useCallback((data) => {
    setMetadata(data)
  }, [])

  /**
   * Clear dataset state entirely.
   */
  const clearDataset = useCallback(() => {
    setFilenameState(null)
    setMetadata(null)
    localStorage.removeItem('dataset_filename')
  }, [])

  const value = {
    filename,
    setFilename,
    metadata,
    setDatasetMetadata,
    clearDataset,
    hasDataset: !!filename,
  }

  return (
    <DatasetContext.Provider value={value}>
      {children}
    </DatasetContext.Provider>
  )
}

/**
 * Hook to consume the DatasetContext.
 * Provides: filename, setFilename, metadata, setDatasetMetadata, clearDataset, hasDataset
 */
export function useDatasetContext() {
  const context = useContext(DatasetContext)
  if (!context) {
    throw new Error('useDatasetContext must be used within a DatasetProvider')
  }
  return context
}

export default DatasetContext
