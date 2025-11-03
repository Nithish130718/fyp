import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import image1 from '../assets/image1.png'
import image2 from '../assets/image2.png'
import image3 from '../assets/image3.png'
import '../App.css'

function Home() {
  const [hoveredCard, setHoveredCard] = useState(null)
  const navigate = useNavigate()

  const handleImageClick = () => {
    navigate('/posture-classification')
  }

  const handleVideoClick = () => {
    navigate('/breath-analysis')
  }

  const handleSleepIndexClick = () => {
    navigate('/sleep-index')
  }

  return (
    <div>
      {/* Main Content */}
      <main className="main-content">
        <div className="welcome-section">
          <h2 className="welcome-title">Welcome to Sleep Quality Assessment</h2>
          <p className="welcome-subtitle">
            Analyze and improve your sleep quality for better health
          </p>
          <p className="instruction-text">Please select a module:</p>
        </div>

        <div className="cards-container">
          {/* Image Classification Card */}
          <div 
            className={`module-card ${hoveredCard === 'image' ? 'hovered' : ''}`}
            onMouseEnter={() => setHoveredCard('image')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <div className="card-illustrations">
              <img src={image1} alt="Sleep posture illustrations" className="card-image" />
            </div>
            <h3 className="card-title">Posture Classification</h3>
            <p className="card-description">
              Performs posture classification using ML and DL model architectures
            </p>
            <button className="card-button" onClick={handleImageClick}>Go To Model</button>
          </div>

          {/* Breath Analysis Card */}
          <div 
            className={`module-card ${hoveredCard === 'breath' ? 'hovered' : ''}`}
            onMouseEnter={() => setHoveredCard('breath')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <div className="card-illustrations">
              <img src={image2} alt="Breath analysis illustrations" className="card-image" />
            </div>
            <h3 className="card-title">Breath Rate Analysis</h3>
            <p className="card-description">
              Analyzes breathing patterns through sensorial data to assess sleep quality
            </p>
            <button className="card-button" onClick={handleVideoClick}>Go To Model</button>
          </div>

          {/* Sleep Fragmentation Index Card */}
          <div 
            className={`module-card ${hoveredCard === 'sleep-index' ? 'hovered' : ''}`}
            onMouseEnter={() => setHoveredCard('sleep-index')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <div className="card-illustrations">
              <img src={image3} alt="Sleep fragmentation index illustrations" className="card-image" />
            </div>
            <h3 className="card-title">Sleep Fragmentation Index</h3>
            <p className="card-description">
              Calculates and analyzes sleep fragmentation patterns using advanced metrics
            </p>
            <button className="card-button" onClick={handleSleepIndexClick}>Go To Model</button>
          </div>
        </div>
      </main>
    </div>
  )
}

export default Home

