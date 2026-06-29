/**
 * ChartRenderer — Recharts wrapper supporting multiple chart types.
 *
 * Renders histogram, scatter, line, bar, pie, box plot, and correlation
 * heatmap visualizations with responsive sizing and tooltip support.
 * (Requirements 9.1, 9.7)
 *
 * Props:
 *   type (string) — one of "histogram", "scatter", "line", "bar", "pie", "box", "heatmap"
 *   data (array | object) — chart data (format varies by chart type)
 *   xKey (string) — key for x-axis values (for bar, line, scatter, histogram)
 *   yKey (string) — key for y-axis values (for scatter, line, bar)
 *   title (string, optional) — chart title
 */
import {
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

const COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe',
  '#00c49f', '#ffbb28', '#ff8042', '#a4de6c', '#d0ed57',
]

/**
 * Maps a correlation value (-1 to +1) to a color on a blue-white-red scale.
 */
function correlationColor(value) {
  const clamped = Math.max(-1, Math.min(1, value))
  if (clamped >= 0) {
    const intensity = Math.round(255 * (1 - clamped))
    return `rgb(255, ${intensity}, ${intensity})`
  } else {
    const intensity = Math.round(255 * (1 + clamped))
    return `rgb(${intensity}, ${intensity}, 255)`
  }
}

function HistogramChart({ data, xKey }) {
  const chartData = data.map((d) => ({
    ...d,
    label: `${Number(d.bin_start).toFixed(1)} – ${Number(d.bin_end).toFixed(1)}`,
  }))

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="label" angle={-45} textAnchor="end" interval={0} height={80} />
        <YAxis />
        <Tooltip />
        <Bar dataKey="count" fill="#8884d8" />
      </BarChart>
    </ResponsiveContainer>
  )
}

function ScatterChartView({ data }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" name="X" type="number" />
        <YAxis dataKey="y" name="Y" type="number" />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Scatter data={data} fill="#8884d8" />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

function LineChartView({ data, xKey, yKey }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey || 'x'} />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey={yKey || 'y'} stroke="#8884d8" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function BarChartView({ data, xKey, yKey }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey || 'category'} angle={-45} textAnchor="end" height={80} />
        <YAxis />
        <Tooltip />
        <Bar dataKey={yKey || 'value'} fill="#82ca9d" />
      </BarChart>
    </ResponsiveContainer>
  )
}

function PieChartView({ data }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={140}
          label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
        >
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  )
}

function BoxPlotView({ data }) {
  const { min, q1, median, q3, max } = data

  return (
    <div className="flex items-center justify-center h-[400px]">
      <div className="bg-white border border-gray-200 rounded-lg p-6 w-full max-w-md shadow-sm">
        <h4 className="text-sm font-medium text-gray-600 mb-4 text-center">Box Plot Statistics</h4>
        <div className="space-y-3">
          <div className="flex justify-between items-center py-1 border-b border-gray-100">
            <span className="text-sm text-gray-500">Max</span>
            <span className="text-sm font-mono font-medium text-gray-800">{Number(max).toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-gray-100">
            <span className="text-sm text-gray-500">Q3 (75th)</span>
            <span className="text-sm font-mono font-medium text-gray-800">{Number(q3).toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-gray-100 bg-blue-50 -mx-2 px-2 rounded">
            <span className="text-sm font-medium text-blue-700">Median</span>
            <span className="text-sm font-mono font-bold text-blue-800">{Number(median).toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-gray-100">
            <span className="text-sm text-gray-500">Q1 (25th)</span>
            <span className="text-sm font-mono font-medium text-gray-800">{Number(q1).toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center py-1">
            <span className="text-sm text-gray-500">Min</span>
            <span className="text-sm font-mono font-medium text-gray-800">{Number(min).toFixed(2)}</span>
          </div>
        </div>
        {/* Visual representation */}
        <div className="mt-6 relative h-8">
          <div className="absolute inset-y-0 left-0 right-0 flex items-center">
            <div className="w-full h-0.5 bg-gray-300 relative">
              {/* Whisker lines and box */}
              <div
                className="absolute top-1/2 -translate-y-1/2 h-6 bg-blue-100 border border-blue-400 rounded-sm"
                style={{
                  left: `${((q1 - min) / (max - min)) * 100}%`,
                  width: `${((q3 - q1) / (max - min)) * 100}%`,
                }}
              />
              {/* Median line */}
              <div
                className="absolute top-1/2 -translate-y-1/2 h-6 w-0.5 bg-blue-700"
                style={{ left: `${((median - min) / (max - min)) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function HeatmapView({ data }) {
  const { columns, matrix } = data

  return (
    <div className="overflow-auto max-h-[400px]">
      <div className="inline-block min-w-full">
        {/* Header row */}
        <div className="flex">
          <div className="w-24 shrink-0" />
          {columns.map((col) => (
            <div
              key={col}
              className="w-16 h-10 flex items-center justify-center text-xs text-gray-600 font-medium truncate px-1"
              title={col}
            >
              {col.length > 6 ? col.slice(0, 6) + '…' : col}
            </div>
          ))}
        </div>
        {/* Matrix rows */}
        {matrix.map((row, rowIdx) => (
          <div key={rowIdx} className="flex">
            <div className="w-24 shrink-0 flex items-center text-xs text-gray-600 font-medium truncate pr-2" title={columns[rowIdx]}>
              {columns[rowIdx].length > 10 ? columns[rowIdx].slice(0, 10) + '…' : columns[rowIdx]}
            </div>
            {row.map((value, colIdx) => (
              <div
                key={colIdx}
                className="w-16 h-12 flex items-center justify-center text-xs font-mono border border-gray-200"
                style={{ backgroundColor: correlationColor(value) }}
                title={`${columns[rowIdx]} × ${columns[colIdx]}: ${Number(value).toFixed(2)}`}
              >
                {Number(value).toFixed(2)}
              </div>
            ))}
          </div>
        ))}
        {/* Color scale legend */}
        <div className="flex items-center gap-2 mt-4 text-xs text-gray-500">
          <span>-1</span>
          <div className="flex h-4 w-48">
            {Array.from({ length: 20 }, (_, i) => {
              const val = -1 + (i / 19) * 2
              return (
                <div
                  key={i}
                  className="flex-1"
                  style={{ backgroundColor: correlationColor(val) }}
                />
              )
            })}
          </div>
          <span>+1</span>
        </div>
      </div>
    </div>
  )
}

function ChartRenderer({ type, data, xKey, yKey, title }) {
  if (!data || (Array.isArray(data) && data.length === 0)) {
    return (
      <div className="flex items-center justify-center h-[400px] text-gray-500">
        <p>No data available for visualization.</p>
      </div>
    )
  }

  const renderChart = () => {
    switch (type) {
      case 'histogram':
        return <HistogramChart data={data} xKey={xKey} />
      case 'scatter':
        return <ScatterChartView data={data} />
      case 'line':
        return <LineChartView data={data} xKey={xKey} yKey={yKey} />
      case 'bar':
        return <BarChartView data={data} xKey={xKey} yKey={yKey} />
      case 'pie':
        return <PieChartView data={data} />
      case 'box':
        return <BoxPlotView data={data} />
      case 'heatmap':
        return <HeatmapView data={data} />
      default:
        return (
          <div className="flex items-center justify-center h-[400px] text-gray-500">
            <p>Unsupported chart type: {type}</p>
          </div>
        )
    }
  }

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-lg font-semibold text-gray-800 mb-3 text-center">{title}</h3>
      )}
      {renderChart()}
    </div>
  )
}

export default ChartRenderer
