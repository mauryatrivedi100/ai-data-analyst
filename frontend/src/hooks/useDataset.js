import { useState, useCallback } from 'react';

/**
 * Custom hook for managing dataset state.
 * Tracks current dataset metadata and preview data after upload.
 */
export function useDataset() {
  const [dataset, setDatasetState] = useState(null);

  /**
   * Update dataset state from upload response.
   * Expected shape: { filename, originalName, rows, columns, columnNames, preview, fileSize }
   */
  const setDataset = useCallback((data) => {
    setDatasetState({
      filename: data.filename ?? null,
      originalName: data.originalName ?? data.original_name ?? null,
      rows: data.rows ?? 0,
      columns: data.columns ?? 0,
      columnNames: data.columnNames ?? data.column_names ?? [],
      preview: data.preview ?? [],
      fileSize: data.fileSize ?? data.file_size ?? '',
    });
  }, []);

  /** Reset dataset state to null. */
  const clearDataset = useCallback(() => {
    setDatasetState(null);
  }, []);

  return {
    dataset,
    setDataset,
    clearDataset,
    hasDataset: dataset !== null,
  };
}

export default useDataset;
