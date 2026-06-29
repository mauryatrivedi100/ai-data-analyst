import api from './api';

/**
 * Upload a CSV file to the backend with progress tracking.
 *
 * @param {File} file - The file to upload
 * @param {function} onProgress - Callback receiving upload progress (0-100)
 * @returns {Promise<object>} Upload response data (filename, rows, columns, preview, file_size, etc.)
 */
export async function uploadFile(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000, // 60s timeout for large file uploads
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onProgress) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percent);
      }
    },
  });

  return response;
}

export default { uploadFile };
