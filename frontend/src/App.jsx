import { Routes, Route, useLocation } from 'react-router-dom'
import { Link } from 'react-router-dom'
import './App.css'
import Home from './pages/Home'
import ImageClassification from './pages/ImageClassification'
import BreathAnalysis from './pages/BreathAnalysis'
import SleepIndex from './pages/SleepIndex'

function App() {
  const location = useLocation()

  const handleNavClick = (path, e) => {
    if (location.pathname === path) {
      // Same route - reload the page to reset to initial state
      e.preventDefault()
      window.location.href = path
    }
  }

  return (
    <>
      {/* Header */}
      <header className="header">
        <div className="header-container">
          <div className="logo-section">
            <Link to="/" style={{ textDecoration: 'none' }}>
              <h1 className="site-title">Sleep Quality Assessment</h1>
            </Link>
          </div>
          <nav className="nav-links">
            <Link 
              to="/posture-classification" 
              className="nav-link" 
              style={{ textDecoration: 'none' }}
              onClick={(e) => handleNavClick('/posture-classification', e)}
            >
              Posture Analysis
            </Link>
            <Link 
              to="/breath-analysis" 
              className="nav-link" 
              style={{ textDecoration: 'none' }}
              onClick={(e) => handleNavClick('/breath-analysis', e)}
            >
              Breath Analysis
            </Link>
            <Link 
              to="/sleep-index" 
              className="nav-link" 
              style={{ textDecoration: 'none' }}
              onClick={(e) => handleNavClick('/sleep-index', e)}
            >
              Sleep Index
            </Link>
          </nav>
        </div>
      </header>

      {/* Routes */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/posture-classification" element={<ImageClassification />} />
        <Route path="/breath-analysis" element={<BreathAnalysis />} />
        <Route path="/sleep-index" element={<SleepIndex />} />
      </Routes>

      {/* Footer */}
      <footer className="footer">
        <p className="footer-text">
          © 2025 SleepQualityAssessment. All rights reserved.
        </p>
      </footer>
    </>
  )
}

export default App
