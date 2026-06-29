import { useState } from 'react'
import { NavLink } from 'react-router-dom'

const navLinks = [
  { to: '/', label: 'Home' },
  { to: '/analysis', label: 'Analysis' },
  { to: '/cleaning', label: 'Cleaning' },
  { to: '/visualization', label: 'Visualization' },
  { to: '/ml', label: 'ML' },
  { to: '/insights', label: 'Insights' },
  { to: '/report', label: 'Report' },
]

function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const linkClasses = ({ isActive }) =>
    `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? 'bg-indigo-100 text-indigo-700'
        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
    }`

  const mobileLinkClasses = ({ isActive }) =>
    `block px-3 py-2 rounded-md text-base font-medium transition-colors ${
      isActive
        ? 'bg-indigo-100 text-indigo-700'
        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
    }`

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          {/* Brand */}
          <NavLink to="/" className="text-xl font-semibold text-gray-800 shrink-0">
            AI Data Analyst
          </NavLink>

          {/* Desktop navigation links */}
          <div className="hidden md:flex md:items-center md:space-x-1">
            {navLinks.map((link) => (
              <NavLink key={link.to} to={link.to} end={link.to === '/'} className={linkClasses}>
                {link.label}
              </NavLink>
            ))}
          </div>

          {/* Mobile menu button */}
          <button
            type="button"
            className="md:hidden inline-flex items-center justify-center p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-expanded={mobileMenuOpen}
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? (
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Mobile navigation menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-gray-200 px-2 pt-2 pb-3 space-y-1">
          {navLinks.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={mobileLinkClasses}
              onClick={() => setMobileMenuOpen(false)}
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  )
}

export default Navbar
