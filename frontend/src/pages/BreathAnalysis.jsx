import { useState } from 'react'
import '../styles/BreathAnalysis.css'
import Loading from '../components/Loading'

function BreathAnalysis() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileName, setFileName] = useState('No file chosen')
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [showResults, setShowResults] = useState(false)

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

  const handleClassify = async () => {
    if (selectedFile) {
      setIsLoading(true)
      setShowResults(false)
      const startTime = Date.now()
      
      const formData = new FormData()
      formData.append('file', selectedFile)
      
      try {
        const response = await fetch('http://localhost:5000/api/breath-analysis', {
          method: 'POST',
          body: formData
        })
        
        const result = await response.json()
        
        // Ensure loading animation shows for at least 3 seconds
        const elapsedTime = Date.now() - startTime
        const remainingTime = Math.max(0, 3000 - elapsedTime)
        
        setTimeout(() => {
          setIsLoading(false)
          
          if (result.status === 'success') {
            setResults(result)
            setShowResults(true)
          } else {
            alert(`Error: ${result.message}`)
          }
        }, Math.max(remainingTime, 6000)) // At least 6 seconds
        
      } catch (error) {
        const elapsedTime = Date.now() - startTime
        const remainingTime = Math.max(0, 3000 - elapsedTime)
        
        setTimeout(() => {
          setIsLoading(false)
          alert(`Failed to analyze file: ${error.message}`)
        }, Math.max(remainingTime, 6000)) // At least 6 seconds
      }
    } else {
      alert('Please select a text file first')
    }
  }


  return (
    <div className="breath-analysis-page">
      {isLoading && <Loading message="Analyzing breath patterns..." />}
      
      <div className={`breath-layout ${showResults ? 'with-results' : ''}`}>
        {/* Left Panel - Upload Section (slides to 30%) */}
        <div className="upload-panel">
          {!showResults && (
            <>
              <h1 className="page-title">Breath Rate Analysis</h1>
              <p className="page-subtitle">
                Analyze and improve your sleep quality for better health
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

            <button className="classify-button" onClick={handleClassify}>
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

            <div className="breathing-rate-display">
              <div className="rate-value">{results.breathing_rate} bpm</div>
              <div className="rate-method">Method: {results.method}</div>
            </div>

            {results.graph_images && results.graph_images.length > 0 && (
              <div className="charts-container">
                {results.graph_images.map((img, index) => (
                  <div key={index} className="chart-wrapper">
                    <img src={`data:image/png;base64,${img}`} alt={`Analysis Result ${index + 1}`} style={{ width: '100%', height: 'auto' }} />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default BreathAnalysis
