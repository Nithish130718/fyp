from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, welch, medfilt, savgol_filter
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
import io, base64, os
import joblib

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def generate_plot(t, sig, baseline, filtered_s, env_s, valid_env, valid_troughs, f, Pxx, dominant_freq):
    """Generate identical visualization as in notebook and return base64 image."""
    images = []
    
    # --- Graph 1: Raw magnitude & baseline ---
    fig1, ax1 = plt.subplots(figsize=(14, 3))
    ax1.plot(t, sig, label='Raw Mag (g)')
    ax1.plot(t, baseline, label='Baseline', alpha=0.7)
    ax1.set_title("Raw Accelerometer Magnitude & Baseline")
    ax1.legend()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
    buf1.seek(0)
    images.append(base64.b64encode(buf1.read()).decode('utf-8'))
    plt.close(fig1)

    # --- Graph 2: Filtered & envelope ---
    fig2, ax2 = plt.subplots(figsize=(14, 3))
    ax2.plot(t, filtered_s, label='Filtered Signal')
    ax2.plot(t, env_s, label='Envelope', alpha=0.8)
    ax2.plot(t[valid_env], env_s[valid_env], 'go', label='Breath Peaks')
    ax2.plot(t[valid_troughs], filtered_s[valid_troughs], 'rv', label='Troughs')
    ax2.set_title("Filtered Signal with Envelope & Detected Breaths")
    ax2.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    buf2.seek(0)
    images.append(base64.b64encode(buf2.read()).decode('utf-8'))
    plt.close(fig2)

    # --- Graph 3: PSD ---
    fig3, ax3 = plt.subplots(figsize=(14, 3))
    ax3.semilogy(f, Pxx + 1e-20)
    ax3.axvline(dominant_freq, color='r', linestyle='--',
                   label=f'Peak: {dominant_freq*60:.1f} bpm')
    ax3.set_title("Power Spectral Density")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.legend()
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', dpi=150, bbox_inches='tight')
    buf3.seek(0)
    images.append(base64.b64encode(buf3.read()).decode('utf-8'))
    plt.close(fig3)
    
    return images


def process_breath_analysis(file_path):
    """Process IMU breathing data and return stats + graph image identical to notebook."""
    try:
        # Load data
        df = pd.read_csv(file_path, sep="\t")
        t_us = df["Timestamp [us]"].values.astype(np.float64)
        ax = df["A_X [mg]"].astype(float).values
        ay = df["A_Y [mg]"].astype(float).values
        az = df["A_Z [mg]"].astype(float).values

        # Time & Sampling Rate
        t = (t_us - t_us[0]) / 1e6
        fs = 1.0 / np.median(np.diff(t))

        # Accel Magnitude
        sig = np.sqrt(ax**2 + ay**2 + az**2) / 1000.0
        sig = np.nan_to_num(sig)

        # Motion Mask
        win_sec = 2.0
        win = max(3, int(win_sec * fs))
        roll_std = pd.Series(sig).rolling(win, center=True).std().bfill().ffill()
        mask = roll_std < 0.06

        # Interpolation
        sig_interp = sig.copy()
        bad = ~mask
        good = ~bad
        if good.sum() > 1:
            sig_interp[bad] = np.interp(np.flatnonzero(bad),
                                        np.flatnonzero(good),
                                        sig[good])

        # Baseline removal
        med_kern = int(fs * 5)
        med_kern += 1 if med_kern % 2 == 0 else 0
        baseline = medfilt(sig_interp, kernel_size=med_kern)
        detrended = sig_interp - baseline

        # Band-pass
        lowcut, highcut = 0.08, 0.6
        sos = butter(4, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band', output='sos')
        filtered = sosfiltfilt(sos, detrended)

        # Smooth
        smooth_win = int(fs * 0.6)
        smooth_win += 1 if smooth_win % 2 == 0 else 0
        filtered_s = savgol_filter(filtered, smooth_win, 2)

        # PSD
        nperseg = min(int(fs * 30), len(filtered_s))
        f, Pxx = welch(filtered_s, fs=fs, nperseg=nperseg)
        band_idx = np.where((f >= lowcut) & (f <= highcut))[0]
        dominant_freq = f[band_idx[np.argmax(Pxx[band_idx])]] if band_idx.size > 0 else np.nan

        # Envelope
        env = np.abs(hilbert(filtered_s))
        env_s = savgol_filter(env, int(fs*1.2)//2*2+1, 2)

        # Peaks / Troughs
        est_period = 1.0 / dominant_freq if np.isfinite(dominant_freq) else 60.0/18.0
        min_dist = int(0.4 * est_period * fs)
        prom = max(0.02 * np.std(env_s), 0.001)
        peaks_env, _ = find_peaks(env_s, distance=min_dist, prominence=prom)
        valid_env = peaks_env[mask[peaks_env]]

        troughs, _ = find_peaks(-filtered_s, distance=min_dist, prominence=prom)
        valid_troughs = troughs[mask[troughs]]

        # Compute RR
        def compute_rr(idx, t):
            if len(idx) < 2:
                return np.nan, np.nan
            ibi = np.diff(t[idx])
            ibi = ibi[(ibi >= 0.5) & (ibi <= 20.0)]
            return 60.0/np.mean(ibi), 60.0/np.median(ibi)

        rr_env_mean, rr_env_median = compute_rr(valid_env, t)
        rr_trough_mean, rr_trough_median = compute_rr(valid_troughs, t)

        if np.isfinite(rr_env_median):
            rr_final, method = rr_env_median, "envelope"
        elif np.isfinite(rr_trough_median):
            rr_final, method = rr_trough_median, "trough"
        elif np.isfinite(dominant_freq):
            rr_final, method = dominant_freq * 60.0, "psd"
        else:
            rr_final, method = np.nan, "none"

        # Manual RR check
        valid_t = t[mask]
        duration_valid = valid_t[-1] - valid_t[0]
        manual_bpm_valid = (len(valid_env) / duration_valid) * 60

        # Generate identical plots
        plot_images = generate_plot(t, sig, baseline, filtered_s, env_s, valid_env,
                                     valid_troughs, f, Pxx, dominant_freq)

        # Response
        return {
            "status": "success",
            "breathing_rate": round(rr_final, 2),
            "method": method,
            "manual_rr": round(manual_bpm_valid, 2),
            "sampling_rate": round(fs, 2),
            "dominant_freq": round(dominant_freq, 3),
            "breath_count": len(valid_env),
            "graph_images": plot_images
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route("/api/breath-analysis", methods=["POST"])
def analyze_breath():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result = process_breath_analysis(file_path)

    os.remove(file_path)
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# Load the trained model at startup
MODEL_PATH = 'posture_classifier_rf.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Loaded posture classification model from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None


def extract_features(window):
    """Extract features from a window of sensor data"""
    features = {}
    axes = {
        "acc": ["A_X [mg]", "A_Y [mg]", "A_Z [mg]"],
        "gyro": ["G_X [dps]", "G_Y [dps]", "G_Z [dps]"]
    }

    for group, cols in axes.items():
        for col in cols:
            if col not in window.columns:
                continue
            data = window[col].values

            # --- Statistical Features ---
            features[f"{col}_mean"] = np.mean(data)
            features[f"{col}_std"] = np.std(data)
            features[f"{col}_min"] = np.min(data)
            features[f"{col}_max"] = np.max(data)
            features[f"{col}_skew"] = skew(data)
            features[f"{col}_kurt"] = kurtosis(data)
            features[f"{col}_range"] = np.max(data) - np.min(data)

            # --- Signal Magnitude Area (SMA) ---
            features[f"{col}_sma"] = np.sum(np.abs(data)) / len(data)

            # --- Frequency Features ---
            fft_vals = np.abs(rfft(data))
            fft_freq = rfftfreq(len(data), d=1/250)  # 250 Hz sampling

            dom_freq = fft_freq[np.argmax(fft_vals)]
            spectral_energy = np.sum(fft_vals**2) / len(fft_vals)

            features[f"{col}_domfreq"] = dom_freq
            features[f"{col}_spec_energy"] = spectral_energy

    return features


def predict_posture_changes(file_path, window_size=500, overlap=0.5):
    """
    Predict posture changes for any new txt file
    Returns: list of (time, posture) tuples
    """
    df = pd.read_csv(file_path, delimiter="\t")
    df['Time_sec'] = df['Timestamp [us]'] / 1_000_000
    
    segments = []
    times = []
    step = int(window_size * (1 - overlap))
    
    for start in range(0, len(df) - window_size, step):
        window = df.iloc[start:start+window_size]
        segments.append(window)
        times.append(window["Time_sec"].mean())
    
    X_new = pd.DataFrame([extract_features(seg) for seg in segments])
    preds = model.predict(X_new)
    
    # Detect posture changes (only record when posture changes)
    changes = []
    prev = None
    for p, t in zip(preds, times):
        if p != prev:
            changes.append((t, p))
            prev = p
    
    return changes


@app.route("/api/posture-classification", methods=["POST"])
def classify_posture():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
        
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        posture_changes = predict_posture_changes(file_path)
        
        # Convert to list of dictionaries for JSON serialization
        results = [{"time": float(t), "posture": p} for t, p in posture_changes]
        
        result = {
            "status": "success",
            "postures": results,
            "total_changes": len(results)
        }
        
        os.remove(file_path)
        return jsonify(result)
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"status": "error", "message": str(e)}), 500


def convert_to_24_hour(time_str, period):
    """Convert 12-hour time with AM/PM to 24-hour format"""
    hour, minute = map(int, time_str.split(':'))
    
    if period.upper() == 'PM' and hour != 12:
        hour += 12
    elif period.upper() == 'AM' and hour == 12:
        hour = 0
    
    return hour, minute


def calculate_sleep_duration(start_time, start_period, wake_time, wake_period):
    """Calculate total sleep duration in hours"""
    start_hour, start_min = convert_to_24_hour(start_time, start_period)
    wake_hour, wake_min = convert_to_24_hour(wake_time, wake_period)
    
    start_total_min = start_hour * 60 + start_min
    wake_total_min = wake_hour * 60 + wake_min
    
    # Handle overnight sleep (e.g., 10 PM to 6 AM)
    if wake_total_min < start_total_min:
        wake_total_min += 24 * 60
    
    duration_min = wake_total_min - start_total_min
    duration_hours = duration_min / 60.0
    
    return duration_hours


def calculate_fsi(posture_changes, total_sleep_time_hr, breathing_rate, age, sex, bmi):
    """Calculate Fragmentation Sleep Index"""
    # Base fragmentation (motion-based)
    FSI_simple = posture_changes / total_sleep_time_hr if total_sleep_time_hr > 0 else 0
    
    # Demographic adjustment
    FSI_adj = FSI_simple * (1 + 0.01 * (bmi - 22)) * (1 + 0.005 * (age - 30))
    if sex.upper() == 'M':
        FSI_adj *= 1.05
    
    # Normalized to 0-100 scale
    FSI_norm = min(FSI_adj / 15, 1.0) * 100
    
    return FSI_simple, FSI_adj, FSI_norm


def get_quality_label(fsi_norm):
    """Get sleep quality interpretation based on FSI"""
    if fsi_norm < 20:
        return "Excellent", "Very low fragmentation, excellent sleep quality"
    elif fsi_norm < 40:
        return "Good", "Low fragmentation, good sleep quality"
    elif fsi_norm < 60:
        return "Fair", "Moderate fragmentation, fair sleep quality"
    elif fsi_norm < 80:
        return "Poor", "High fragmentation, poor sleep quality"
    else:
        return "Very Poor", "Very high fragmentation, very poor sleep quality"


@app.route("/api/sleep-index", methods=["POST"])
def calculate_sleep_index():
    """Calculate Fragmentation Sleep Index"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Get form data
    sleep_start = request.form.get('sleep_start', '')
    sleep_start_period = request.form.get('sleep_start_period', 'PM')
    wake_time = request.form.get('wake_time', '')
    wake_period = request.form.get('wake_period', 'AM')
    sex = request.form.get('sex', '')
    age_str = request.form.get('age', '')
    weight_str = request.form.get('weight', '')
    height_str = request.form.get('height', '')
    
    # Validate required fields
    if not sleep_start or not wake_time or not sex or not age_str or not weight_str or not height_str:
        return jsonify({"status": "error", "message": "All fields are required"}), 400
    
    # Convert to appropriate types
    try:
        age = int(age_str)
        weight = float(weight_str)
        height_cm = float(height_str)
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid input values"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Validate file has required columns
        required_cols = ["Timestamp [us]", "A_X [mg]", "A_Y [mg]", "A_Z [mg]"]
        
        # Calculate total sleep time
        total_sleep_time_hr = calculate_sleep_duration(
            sleep_start, sleep_start_period, wake_time, wake_period
        )
        
        if total_sleep_time_hr <= 0:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": "Invalid sleep time range"}), 400
        
        # Calculate BMI (convert height to meters)
        height_m = height_cm / 100.0
        bmi = weight / (height_m ** 2)
        
        # Analyze the file to get posture changes and breathing rate
        df = pd.read_csv(file_path, delimiter="\t")
        
        # Validate file has required columns
        if not all(col in df.columns for col in ["Timestamp [us]", "A_X [mg]", "A_Y [mg]", "A_Z [mg]"]):
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": "File missing required columns"}), 400
        
        # Analyze posture changes from the file
        posture_changes = 0
        breathing_rate = 14.5  # Default
        
        # Get actual posture changes using the trained model
        if model and 'A_X [mg]' in df.columns and 'A_Y [mg]' in df.columns and 'A_Z [mg]' in df.columns:
            # Use the posture classification model to get actual posture changes
            try:
                # Segment the data
                window_size = 500
                overlap = 0.5
                step = int(window_size * (1 - overlap))
                
                segments = []
                times = []
                
                for start in range(0, len(df) - window_size, step):
                    window = df.iloc[start:start+window_size]
                    segments.append(window)
                    times.append(window["Timestamp [us]"].mean() / 1_000_000)
                
                # Extract features and predict
                X_new = pd.DataFrame([extract_features(seg) for seg in segments])
                preds = model.predict(X_new)
                
                # Count actual posture changes (only when posture actually changes)
                changes = 0
                prev = None
                for p in preds:
                    if p != prev:
                        changes += 1
                        prev = p
                
                posture_changes = int(changes)
                
                # Get actual breathing rate from the file
                if len(df) > 0:
                    try:
                        # Use the breath analysis function to get real breathing rate
                        breath_result = process_breath_analysis(file_path)
                        if isinstance(breath_result, dict) and breath_result.get("status") == "success" and "breathing_rate" in breath_result:
                            breathing_rate = float(breath_result["breathing_rate"])
                        else:
                            breathing_rate = float(14.5)  # Default if analysis fails
                            print("Breath analysis failed, using default")
                    except Exception as breath_error:
                        print(f"Error in breath analysis: {breath_error}")
                        import traceback
                        traceback.print_exc()
                        breathing_rate = float(14.5)  # Default on error
                        
            except Exception as e:
                print(f"Error in posture analysis: {e}")
                # Fallback
                posture_changes = 15  # Default reasonable value
                breathing_rate = float(14.5)
        else:
            # If columns missing or model not loaded, use defaults
            print("Warning: Using default values")
            posture_changes = 15
            breathing_rate = float(14.5)
        
        # Calculate FSI
        FSI_simple, FSI_adj, FSI_norm = calculate_fsi(
            posture_changes, total_sleep_time_hr, breathing_rate, age, sex, bmi
        )
        
        # Get quality interpretation
        quality, interpretation = get_quality_label(FSI_norm)
        
        result = {
            "status": "success",
            "fsi_norm": float(FSI_norm),  # Convert to Python float
            "fsi_adj": float(FSI_adj),   # Convert to Python float
            "fsi_simple": float(FSI_simple),  # Convert to Python float
            "total_sleep_time_hr": float(total_sleep_time_hr),  # Convert to Python float
            "posture_changes": int(posture_changes),  # Convert to Python int
            "breathing_rate": float(breathing_rate),  # Convert to Python float
            "bmi": float(bmi),  # Convert to Python float
            "quality": str(quality),  # Ensure string
            "interpretation": str(interpretation)  # Ensure string
        }
        
        os.remove(file_path)
        return jsonify(result)
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Error in sleep-index endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
