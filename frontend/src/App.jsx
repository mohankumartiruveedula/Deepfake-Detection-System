import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud } from 'lucide-react';

function App() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUri, setPreviewUri] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const processFile = async (selectedFile) => {
    if (!selectedFile) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/avi', 'video/quicktime'];
    if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(mkv|webm)$/i)) {
      setError('Unsupported file format. Please upload JPG, PNG, WEBP, MP4, AVI, MKV, or WEBM.');
      return;
    }

    setFile(selectedFile);
    setError(null);
    setPreviewUri(URL.createObjectURL(selectedFile));
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to backend. Make sure the server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const resetState = () => {
    setFile(null);
    setPreviewUri(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  useEffect(() => {
    return () => {
      if (previewUri) URL.revokeObjectURL(previewUri);
    };
  }, [previewUri]);

  // CSS class for verdict box: REAL → 'real', FAKE → 'fake', UNCERTAIN → 'uncertain'
  const verdictClass = result ? result.label.toLowerCase() : '';

  return (
    <div className="app-container">
      <div className="header">
        <h1>Deepfake Detector</h1>
        <p>EfficientNet-B4 · Grad-CAM heatmaps · Trained on DF40 (40 manipulation methods)</p>
      </div>

      <div className="glass-panel">

        {/* ── Upload zone ── */}
        {!loading && !result && !error && (
          <div
            id="upload-zone"
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            <UploadCloud className="upload-icon" />
            <p className="upload-text">Drag &amp; drop an image or video</p>
            <p className="upload-subtext">Supports JPG, PNG, WEBP, MP4, AVI, MKV, WEBM · Max 50 MB</p>
            <input
              ref={inputRef}
              id="file-input"
              type="file"
              className="file-input"
              accept="image/jpeg,image/png,image/webp,video/mp4,video/avi,video/webm,video/x-matroska,video/quicktime"
              onChange={handleChange}
            />
          </div>
        )}

        {/* ── Loading spinner ── */}
        {loading && (
          <div className="loader-container" id="loader">
            <div className="spinner"></div>
            <p className="pulsing-text">Analyzing media artifacts…</p>
            {file && file.type.includes('video') && (
              <p className="detail-text">Videos are sampled every 8th frame — this may take a moment</p>
            )}
          </div>
        )}

        {/* ── Error state ── */}
        {error && !loading && (
          <div className="verdict-box fake" id="error-box" style={{ marginTop: '1rem' }}>
            <h3 style={{ color: 'var(--fake-color)', marginBottom: '0.5rem' }}>Error</h3>
            <p className="detail-text">{error}</p>
            <button className="reset-btn" id="try-again-btn" style={{ marginTop: '1.25rem' }} onClick={resetState}>
              Try Again
            </button>
          </div>
        )}

        {/* ── Results ── */}
        {result && !loading && (
          <div className="results-view" id="results-view">

            {/* Preview row: original + Grad-CAM heatmap side-by-side */}
            <div className="preview-row">
              <div className="preview-container" id="preview-original">
                <p className="preview-label">Original Input</p>
                {file?.type.includes('video') ? (
                  <video src={previewUri} className="preview-media" controls muted />
                ) : (
                  <img src={previewUri} alt="Uploaded preview" className="preview-media" />
                )}
              </div>

              {result.heatmap_overlay_base64 && (
                <div className="preview-container" id="preview-heatmap">
                  <p className="preview-label">Grad-CAM Heatmap</p>
                  <img
                    src={`data:image/png;base64,${result.heatmap_overlay_base64}`}
                    alt="Grad-CAM heatmap overlay"
                    className="preview-media"
                  />
                </div>
              )}
            </div>

            {/* Verdict card */}
            <div className={`verdict-box ${verdictClass}`} id="verdict-box">
              <div className="verdict-label" id="verdict-label">{result.label}</div>
              <p className="detail-text">File: <strong style={{ color: 'var(--text-main)' }}>{result.filename}</strong></p>

              <div className="confidence-bar-bg">
                <div
                  className="confidence-bar-fill"
                  style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                ></div>
              </div>

              <div className="stats-row">
                <span className="detail-text">
                  Confidence: <strong style={{ color: 'var(--text-main)' }}>{(result.confidence * 100).toFixed(2)}%</strong>
                </span>
                <span className="detail-text">
                  Raw fake probability: <strong style={{ color: 'var(--text-main)' }}>{result.fake_prob.toFixed(4)}</strong>
                </span>
              </div>

              {result.label === 'UNCERTAIN' && (
                <p className="detail-text uncertain-note">
                  ⚠️ The model&apos;s confidence is low (probability in 0.35–0.50 range). Human review is recommended.
                </p>
              )}
            </div>

            <button className="reset-btn" id="reset-btn" onClick={resetState}>
              Analyze Another File
            </button>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
