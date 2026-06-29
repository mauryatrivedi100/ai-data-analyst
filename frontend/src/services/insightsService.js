import { aiApi } from './api'

/**
 * insightsService — API calls for AI-powered dataset insights.
 * Uses the aiApi instance (30-second timeout) for longer AI processing times.
 * (Requirements 13.1–13.5)
 */

/**
 * Generate AI-powered insights for the uploaded dataset.
 *
 * @param {string} filename — the stored dataset filename
 * @returns {Promise<{overview: string, observations: string, business_insights: string, risks: string, recommendations: string}>}
 */
export function generateInsights(filename) {
  return aiApi.post('/generate-insights', { filename })
}
