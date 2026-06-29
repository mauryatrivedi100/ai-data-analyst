function DataTable({ data, columns, maxRows = 20 }) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500 bg-gray-50 rounded-lg border border-gray-200">
        No data available
      </div>
    )
  }

  const columnHeaders = columns || Object.keys(data[0])
  const displayedRows = data.slice(0, maxRows)
  const isTruncated = data.length > maxRows

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto overflow-y-auto max-h-96 border border-gray-200 rounded-lg">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              {columnHeaders.map((col) => (
                <th
                  key={col}
                  className="px-4 py-3 text-left font-semibold text-gray-700 whitespace-nowrap border-b border-gray-200"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {displayedRows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className={`hover:bg-indigo-50 transition-colors ${
                  rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                }`}
              >
                {columnHeaders.map((col) => (
                  <td
                    key={`${rowIndex}-${col}`}
                    className="px-4 py-2 whitespace-nowrap text-gray-600"
                  >
                    {row[col] != null ? String(row[col]) : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {isTruncated && (
        <p className="text-sm text-gray-500 text-right">
          Showing {maxRows} of {data.length} rows
        </p>
      )}
    </div>
  )
}

export default DataTable
