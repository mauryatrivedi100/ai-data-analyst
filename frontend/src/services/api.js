import axios from 'axios';

/**
 * Base Axios instance for communicating with the Flask backend.
 * Default timeout: 30 seconds for regular requests.
 */
const api = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Request interceptor — passes through requests as-is.
 * Can be extended later for auth tokens or request logging.
 */
api.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

/**
 * Response interceptor:
 * - On success: unwraps response.data so callers get the payload directly.
 * - On error: transforms into a user-friendly {message, code} object.
 */
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const transformed = transformError(error);
    return Promise.reject(transformed);
  }
);

/**
 * Transforms Axios errors into a consistent {message, code} shape.
 * All messages are user-friendly and non-technical (Requirements 16.3).
 */
function transformError(error) {
  // Network error (no response received)
  if (!error.response && error.request) {
    if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      return { message: 'The request took too long. Please try again.', code: 'TIMEOUT' };
    }
    return {
      message: 'Unable to reach the server. Please check your connection and try again.',
      code: 'NETWORK_ERROR',
    };
  }

  // Server responded with an error status
  if (error.response) {
    const { status, data } = error.response;
    const serverMessage = data?.error;

    // Map HTTP statuses to friendly messages when no server message exists
    const fallbackMessages = {
      400: 'The request could not be processed. Please check your input and try again.',
      404: 'The requested resource was not found. Please try uploading your dataset again.',
      409: 'This operation conflicts with the current state of your data.',
      422: 'The data could not be processed. Please verify your input.',
      500: 'Something went wrong on our end. Please try again.',
      502: 'The service is temporarily unavailable. Please try again later.',
      504: 'The service did not respond in time. Please try again.',
    };

    return {
      message: serverMessage || fallbackMessages[status] || 'Something went wrong. Please try again.',
      code: data?.code || `HTTP_${status}`,
    };
  }

  // Fallback for unexpected error shapes
  return { message: 'Something went wrong. Please try again.', code: 'UNKNOWN' };
}

/**
 * Axios instance configured for AI-related requests (local Ollama LLM).
 * Timeout set to 120 seconds to accommodate local model inference times.
 */
const aiApi = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

aiApi.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

aiApi.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const transformed = transformError(error);
    return Promise.reject(transformed);
  }
);

export { aiApi };
export default api;
