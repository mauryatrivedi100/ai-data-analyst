import { Outlet } from 'react-router-dom'
import Navbar from './Navbar'

function Layout() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Persistent navigation */}
      <Navbar />

      {/* Main content area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout
