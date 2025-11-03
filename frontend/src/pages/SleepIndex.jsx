import { useState } from 'react'
import '../styles/SleepIndex.css'
import Loading from '../components/Loading'

function SleepIndex() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileName, setFileName] = useState('No file chosen')
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [showResults, setShowResults] = useState(false)
  
  // Form inputs
  const [sleepStart, setSleepStart] = useState('')
  const [sleepStartPeriod, setSleepStartPeriod] = useState('PM')
  const [wakeTime, setWakeTime] = useState('')
  const [wakePeriod, setWakePeriod] = useState('AM')
  const [sex, setSex] = useState('')
  const [age, setAge] = useState('')
  const [weight, setWeight] = useState('')
  const [height, setHeight] = useState('')

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setFileName(file.name)
    } else {
      setSelectedFile(null)
      setFileName('No file chosen')
    }
  }

  const handleDownload = () => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile)
      const a = document.createElement('a')
      a.href = url
      a.download = selectedFile.name
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      alert('Please select a text file first')
      return
    }
    
    // Validate required fields
    if (!sleepStart || !wakeTime || !sex || !age || !weight || !height) {
      alert('Please fill in all required fields')
      return
    }
    
    setIsLoading(true)
    setShowResults(false)
    const startTime = Date.now()
    
    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('sleep_start', sleepStart)
    formData.append('sleep_start_period', sleepStartPeriod)
    formData.append('wake_time', wakeTime)
    formData.append('wake_period', wakePeriod)
    formData.append('sex', sex)
    formData.append('age', age)
    formData.append('weight', weight)
    formData.append('height', height)
    
    try {
        const response = await fetch('http://localhost:5000/api/sleep-index', {
          method: 'POST',
          body: formData
        })
        
        const result = await response.json()
        
        // Ensure loading animation shows for at least 5 seconds
        const elapsedTime = Date.now() - startTime
        const remainingTime = Math.max(0, 5000 - elapsedTime)
        
        setTimeout(() => {
          setIsLoading(false)
          
          if (result.status === 'success') {
            setResults(result)
            setShowResults(true)
          } else {
            alert(`Error: ${result.message}`)
          }
        }, remainingTime)
        
      } catch (error) {
        const elapsedTime = Date.now() - startTime
        const remainingTime = Math.max(0, 5000 - elapsedTime)
        
        console.error('Error details:', error)
        
        setTimeout(() => {
          setIsLoading(false)
          alert(`Failed to analyze file: ${error.message}. Please check console for details.`)
        }, remainingTime)
      }
  }

  return (
    <div className="sleep-index-page">
      {isLoading && <Loading message="Analyzing sleep fragmentation..." />}
      
      <div className={`sleep-index-layout ${showResults ? 'with-results' : ''}`}>
        {/* Left Panel - Upload Section (slides to 30%) */}
        <div className="upload-panel">
          {!showResults && (
            <>
              <h1 className="page-title">Sleep Fragmentation Index</h1>
              <p className="page-subtitle">
                Analyze and assess your sleep fragmentation patterns
              </p>
            </>
          )}

          <div className="upload-section">
            <h2 className="upload-title">Upload File</h2>
            
            <div className="file-input-wrapper">
              <label htmlFor="file-upload" className="upload-label">Upload text file</label>
              <div className="file-input-container">
                <input
                  type="file"
                  id="file-upload"
                  className="file-input"
                  onChange={handleFileChange}
                  accept=".txt"
                />
                <label htmlFor="file-upload" className="file-input-label">
                  <span className="file-button">Choose File</span>
                  <span className="file-name">{fileName}</span>
                </label>
              </div>
            </div>

            <h2 className="upload-title" style={{ marginTop: '1.5rem' }}>Sleep Information</h2>
            
            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
                <label className="form-label">Sleep Start Time</label>
                <div className="time-input-group">
                  <input 
                    type="time" 
                    className="time-input"
                    value={sleepStart}
                    onChange={(e) => setSleepStart(e.target.value)}
                  />
                  <select 
                    className="period-select"
                    value={sleepStartPeriod}
                    onChange={(e) => setSleepStartPeriod(e.target.value)}
                  >
                    <option value="AM">AM</option>
                    <option value="PM">PM</option>
                  </select>
                </div>
              </div>

              <div className="form-group" style={{ flex: 1 }}>
                <label className="form-label">Wake Time</label>
                <div className="time-input-group">
                  <input 
                    type="time" 
                    className="time-input"
                    value={wakeTime}
                    onChange={(e) => setWakeTime(e.target.value)}
                  />
                  <select 
                    className="period-select"
                    value={wakePeriod}
                    onChange={(e) => setWakePeriod(e.target.value)}
                  >
                    <option value="AM">AM</option>
                    <option value="PM">PM</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Sex</label>
                <select 
                  className="form-input"
                  value={sex}
                  onChange={(e) => setSex(e.target.value)}
                >
                  <option value="">Select sex</option>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Age</label>
                <input 
                  type="number" 
                  className="form-input"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  min="1"
                  max="120"
                  placeholder="Enter age"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Height (cm)</label>
                <input 
                  type="number" 
                  className="form-input"
                  value={height}
                  onChange={(e) => setHeight(e.target.value)}
                  min="1"
                  max="250"
                  placeholder="Enter height"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Weight (kg)</label>
                <input 
                  type="number" 
                  className="form-input"
                  value={weight}
                  onChange={(e) => setWeight(e.target.value)}
                  min="1"
                  max="200"
                  placeholder="Enter weight"
                />
              </div>
            </div>

            <button className="classify-button" onClick={handleAnalyze}>
              Analyze
            </button>
          </div>
        </div>

        {/* Right Panel - Results (slides in 70%) */}
        {showResults && results && (
          <div className="results-panel">
            <div className="file-display" onClick={handleDownload}>
              <span className="file-icon">📄</span>
              <span className="file-name-text">{fileName}</span>
              <span className="download-hint">Click to download</span>
            </div>

            <div className="sleep-index-display">
              <div className="index-header">
                <h3>Sleep Fragmentation Index</h3>
                <span className="index-value">{results.fsi_norm?.toFixed(1) || 'N/A'}</span>
              </div>
              
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Total Sleep Time</div>
                  <div className="metric-value">{results.total_sleep_time_hr ? `${results.total_sleep_time_hr.toFixed(2)} hrs` : 'N/A'}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Posture Changes</div>
                  <div className="metric-value">{results.posture_changes || 'N/A'}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Breathing Rate</div>
                  <div className="metric-value">{results.breathing_rate ? `${results.breathing_rate.toFixed(1)} bpm` : 'N/A'}</div>
                </div>
              </div>

              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">BMI</div>
                  <div className="metric-value">{results.bmi?.toFixed(1) || 'N/A'}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">FSI (Adjusted)</div>
                  <div className="metric-value">{results.fsi_adj?.toFixed(2) || 'N/A'}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Sleep Quality</div>
                  <div className="metric-value">{results.quality || 'N/A'}</div>
                </div>
              </div>

              {results.interpretation && (
                <div className="interpretation-box">
                  <h4>Interpretation</h4>
                  <p>{results.interpretation}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default SleepIndex

