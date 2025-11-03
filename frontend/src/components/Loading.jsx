import Lottie from 'lottie-react'
import loadingAnimation from '../assets/loading.json'
import '../styles/Loading.css'

function Loading({ message = 'Processing...' }) {
  return (
    <div className="loading-container">
      <div className="loading-content">
        <Lottie 
          animationData={loadingAnimation} 
          loop={true}
          style={{ width: 300, height: 300 }}
        />
        <p className="loading-message">{message}</p>
      </div>
    </div>
  )
}

export default Loading

