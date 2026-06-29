import axios from 'axios'

/**
 * reportService — API calls for PDF report download.
 * Uses a raw axios instance (bypassing the default response interceptor)
 * because the response is a binary blob, not JSON.
 * (Requirements 14.1, 14.4, 14.5)
 */

const reportApi = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 60000, // Reports may take time to generate
})

/**
 * Download the generated PDF report for the given dataset.
 * Triggers a browser download of the returned PDF blob.
 *
 * @param {string} filename — the stored dataset filename
 * @returns {Promise<string>} — the downloaded file name
 */
export async function downloadReport(filename) {
  const response = await reportApi.get('/download-report', {
    params: { filename },
    responseType: 'blob',
  })

  // Extract filename from Content-Disposition header or use default
  const contentDisposition = response.headers['content-disposition']
  let downloadName = `report_${filename.replace('.csv', '')}.pdf`
  if (contentDisposition) {
    const match = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
    if (match && match[1]) {
      downloadName = match[1].replace(/['"]/g, '')
    }
  }

  // Create blob URL and trigger browser download
  const blob = new Blob([response.data], { type: 'application/pdf' })
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', downloadName)
  document.body.appendChild(link)
  link.click()

  // Clean up
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)

  return downloadName
}
