import api from './api'

/**
 * mlService — API calls for ML model training and feature importance.
 * (Requirements 10.1–10.6, 11.1–11.6, 12.1–12.4)
 */

/**
 * Train a machine learning model on the uploaded dataset.
 *
 * @param {string} filename — the stored dataset filename
 * @param {string} target — the target column name
 * @param {string} algorithm — algorithm identifier (e.g., "random_forest")
 * @param {string} taskType — "classification" or "regression"
 * @returns {Promise<object>} — metrics, confusion_matrix, predictions, feature_importance
 */
export function trainModel(filename, target, algorithm, taskType) {
  return api.post('/train', {
    filename,
    target,
    algorithm,
    task_type: taskType,
  })
}

/**
 * Get feature importance scores for the last trained model.
 *
 * @param {string} filename — the stored dataset filename
 * @returns {Promise<object>} — { features: [{name, importance}], available, message }
 */
export function getFeatureImportance(filename) {
  return api.get('/feature-importance', { params: { filename } })
}
