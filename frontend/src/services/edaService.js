import api from './api';

/**
 * Fetch visualization chart data from the backend.
 *
 * @param {string} filename - The dataset filename on the server.
 * @param {string} type - Chart type (histogram, scatter, line, bar, pie, box, heatmap).
 * @param {string} [x] - Column name for x-axis.
 * @param {string} [y] - Column name for y-axis.
 * @returns {Promise<{chart_type: string, data: Array|Object}>}
 */
export async function getVisualization(filename, type, x, y) {
  const params = { filename, type };
  if (x) params.x = x;
  if (y) params.y = y;
  return api.get('/visualizations', { params });
}
