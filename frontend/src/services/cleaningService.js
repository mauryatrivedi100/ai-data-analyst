import api from './api';

/**
 * Generic cleaning operation — POST /clean with {filename, operation, params}.
 * @param {string} filename - The dataset filename on the server.
 * @param {string} operation - The cleaning operation identifier.
 * @param {object} [params={}] - Additional parameters for the operation.
 * @returns {Promise<object>} The server response with success, summary, dataset_info.
 */
export async function cleanDataset(filename, operation, params = {}) {
  return api.post('/clean', { filename, operation, params });
}

/**
 * Remove all rows that contain any missing values.
 */
export async function removeMissingRows(filename) {
  return cleanDataset(filename, 'remove_missing');
}

/**
 * Fill missing values in a column with the column mean.
 */
export async function fillMean(filename, column) {
  return cleanDataset(filename, 'fill_mean', { column });
}

/**
 * Fill missing values in a column with the column median.
 */
export async function fillMedian(filename, column) {
  return cleanDataset(filename, 'fill_median', { column });
}

/**
 * Fill missing values in a column with the column mode.
 */
export async function fillMode(filename, column) {
  return cleanDataset(filename, 'fill_mode', { column });
}

/**
 * Remove duplicate rows from the dataset.
 */
export async function removeDuplicates(filename) {
  return cleanDataset(filename, 'remove_duplicates');
}

/**
 * Detect outliers in a numerical column using IQR method.
 */
export async function detectOutliers(filename, column) {
  return cleanDataset(filename, 'detect_outliers', { column });
}

/**
 * Remove outlier rows from a numerical column using IQR method.
 */
export async function removeOutliers(filename, column) {
  return cleanDataset(filename, 'remove_outliers', { column });
}

/**
 * Drop specified columns from the dataset.
 * @param {string} filename
 * @param {string[]} columns - Array of column names to drop.
 */
export async function dropColumns(filename, columns) {
  return cleanDataset(filename, 'drop_columns', { columns });
}

/**
 * Rename a column.
 * @param {string} filename
 * @param {string} column - Current column name.
 * @param {string} newName - New column name.
 */
export async function renameColumn(filename, column, newName) {
  return cleanDataset(filename, 'rename_column', { column, new_name: newName });
}

/**
 * Convert a column to a target data type.
 * @param {string} filename
 * @param {string} column
 * @param {string} targetType - Target type (e.g., "int", "float", "string", "datetime").
 */
export async function convertType(filename, column, targetType) {
  return cleanDataset(filename, 'convert_type', { column, target_type: targetType });
}

/**
 * Apply label encoding to a categorical column.
 */
export async function labelEncode(filename, column) {
  return cleanDataset(filename, 'label_encode', { column });
}

/**
 * Apply one-hot encoding to a categorical column.
 */
export async function oneHotEncode(filename, column) {
  return cleanDataset(filename, 'one_hot_encode', { column });
}

/**
 * Apply standard scaling (z-score normalization) to a numerical column.
 */
export async function standardScale(filename, column) {
  return cleanDataset(filename, 'standard_scale', { column });
}

/**
 * Apply min-max scaling to a numerical column.
 */
export async function minMaxScale(filename, column) {
  return cleanDataset(filename, 'min_max_scale', { column });
}

/**
 * Download the cleaned dataset as a CSV file.
 * Triggers a browser download using a blob response.
 * @param {string} filename - The cleaned dataset filename.
 */
export async function downloadCleaned(filename) {
  const response = await api.get('/download-cleaned', {
    params: { filename },
    responseType: 'blob',
    // Bypass the response interceptor's data unwrap since we need the full response
    transformResponse: undefined,
  });

  // The interceptor already unwraps response.data, but with responseType: 'blob'
  // we need to handle the download manually
  const blob = response instanceof Blob ? response : new Blob([response]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `cleaned_${filename}`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}
