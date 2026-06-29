import { BrowserRouter, Routes, Route } from 'react-router-dom'
import ErrorBoundary from './components/ErrorBoundary'
import { DatasetProvider } from './contexts/DatasetContext'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import AnalysisDashboardPage from './pages/AnalysisDashboardPage'
import CleaningPage from './pages/CleaningPage'
import VisualizationPage from './pages/VisualizationPage'
import MLPage from './pages/MLPage'
import InsightsPage from './pages/InsightsPage'
import ReportPage from './pages/ReportPage'

function App() {
  return (
    <ErrorBoundary>
      <DatasetProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<HomePage />} />
              <Route path="/analysis" element={<AnalysisDashboardPage />} />
              <Route path="/cleaning" element={<CleaningPage />} />
              <Route path="/visualization" element={<VisualizationPage />} />
              <Route path="/ml" element={<MLPage />} />
              <Route path="/insights" element={<InsightsPage />} />
              <Route path="/report" element={<ReportPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </DatasetProvider>
    </ErrorBoundary>
  )
}

export default App
