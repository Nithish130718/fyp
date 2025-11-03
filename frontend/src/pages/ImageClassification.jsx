import { useState } from 'react'
import '../styles/ImageClassification.css'
import Loading from '../components/Loading'

function ImageClassification() {
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
        const response = await fetch('http://localhost:5000/api/posture-classification', {
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
        
        setTimeout(() => {
          setIsLoading(false)
          alert(`Failed to classify file: ${error.message}`)
        }, remainingTime)
      }
    } else {
      alert('Please select a text file first')
    }
  }

  return (
    <div className="classification-page">
      {isLoading && <Loading message="Classifying your file..." />}
      
      <div className={`classification-layout ${showResults ? 'with-results' : ''}`}>
        {/* Left Panel - Upload Section (slides to 30%) */}
        <div className="upload-panel">
          {!showResults && (
            <>
              <h1 className="page-title">Sleep Posture Classification</h1>
              <p className="page-subtitle">
                Analyze and improve your sleep posture for better health
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
              Classify
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

            <div className="postures-display">
              <div className="postures-header">
                <h3>Detected Posture Changes</h3>
                <span className="total-changes">{results.total_changes} changes detected</span>
              </div>
              
              <div className="postures-list">
                {results.postures.map((item, index) => {
                  const nextTime = index < results.postures.length - 1 
                    ? results.postures[index + 1].time.toFixed(2)
                    : 'END';
                  const timeDisplay = index < results.postures.length - 1
                    ? `${item.time.toFixed(2)}s - ${nextTime}s`
                    : `${item.time.toFixed(2)}s - ${nextTime}`;
                  
                  return (
                    <div key={index} className="posture-item">
                      <span className="posture-time">{timeDisplay}</span>
                      <span className="posture-label">{item.posture}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ImageClassification

