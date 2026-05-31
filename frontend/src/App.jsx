import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Shield, Cpu, Zap, RotateCcw, AlertTriangle } from 'lucide-react';

function App() {
  const [dragActive, setDragActive]   = useState(false);
  const [file, setFile]               = useState(null);
  const [previewUri, setPreviewUri]   = useState(null);
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [scanLine, setScanLine]       = useState('INITIALIZING...');
  const inputRef = useRef(null);

  // Cycling scan log messages while analyzing
  useEffect(() => {
    if (!loading) return;
    const msgs = [
      'DETECTING FACE BOUNDARIES...',
      'RUNNING EFFICIENTNET-B4...',
      'APPLYING SRM FILTER...',
      'COMPUTING GRAD-CAM...',
      'ANALYZING FREQUENCY ARTIFACTS...',
      'CROSS-REFERENCING DF40 PATTERNS...',
      'FINALIZING VERDICT...',
    ];
    let i = 0;
    const interval = setInterval(() => {
      setScanLine(msgs[i % msgs.length]);
      i++;
    }, 900);
    return () => clearInterval(interval);
  }, [loading]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const processFile = async (selectedFile) => {
    if (!selectedFile) return;
    const validTypes = ['image/jpeg','image/png','image/webp','video/mp4','video/avi','video/quicktime'];
    if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(mkv|webm)$/i)) {
      setError('UNSUPPORTED FORMAT. UPLOAD JPG, PNG, WEBP, MP4, AVI, MKV, OR WEBM.');
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
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/detect`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `SERVER ERROR: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'CONNECTION FAILED. ENSURE BACKEND IS RUNNING ON PORT 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0]);
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files?.[0]) processFile(e.target.files[0]);
  };

  const resetState = () => {
    setFile(null);
    setPreviewUri(null);
    setResult(null);
    setError(null);
    setScanLine('INITIALIZING...');
    if (inputRef.current) inputRef.current.value = '';
  };

  useEffect(() => {
    return () => { if (previewUri) URL.revokeObjectURL(previewUri); };
  }, [previewUri]);

  const verdictClass = result ? result.label.toLowerCase() : '';

  const verdictIcon = {
    real:      <Shield size={20} strokeWidth={1.5} />,
    fake:      <AlertTriangle size={20} strokeWidth={1.5} />,
    uncertain: <Zap size={20} strokeWidth={1.5} />,
  };

  return (
    <div className="app-container">

      {/* ── HEADER ───────────────────────────────────────────── */}
      <div className="header">
        <div className="header-status-bar">
          <div className="status-dot" />
          <span>SYSTEM ONLINE</span>
          <span style={{ color: 'var(--accent)', textShadow: '0 0 6px rgba(0,255,136,0.5)' }}>MODEL: EFFICIENTNET-B4</span>
          <div className="status-dot" />
        </div>

        <h1>DEEPFAKE<br />DETECTOR</h1>

        <div className="header-subtitle">
          <span className="header-badge">DF40 Dataset</span>
          <span style={{ color: 'var(--border)' }}>|</span>
          <span className="header-badge">Grad-CAM Heatmaps</span>
          <span style={{ color: 'var(--border)' }}>|</span>
          <span className="header-badge">40 Manipulation Methods</span>
        </div>

        {/* ── Scope Warning Caption ── */}
        <div className="scope-warning">
          <span className="scope-warning-icon">⚠</span>
          <span>DESIGNED FOR PORTRAIT HUMAN FACE DEEPFAKES ONLY — FULL-BODY, OBJECT, OR NON-FACE IMAGES MAY YIELD INACCURATE RESULTS</span>
        </div>
      </div>

      {/* ── MAIN PANEL ───────────────────────────────────────── */}
      <div className="cyber-panel">

        {/* Terminal title bar */}
        <div className="panel-header">
          <div className="panel-dot red" />
          <div className="panel-dot yellow" />
          <div className="panel-dot green" />
          <span style={{ marginLeft: '0.5rem' }}>terminal@deepfake-detector:~$</span>
          <span className="panel-title"><Cpu size={12} strokeWidth={1.5} style={{ display: 'inline', marginRight: '4px' }} />INFERENCE ENGINE v2.0</span>
        </div>

        <div className="panel-body">

          {/* ── UPLOAD ZONE ────────────────────────────────── */}
          {!loading && !result && !error && (
            <div
              id="upload-zone"
              className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Upload image or video for deepfake detection"
              onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
            >
              {/* Corner markers */}
              <div className="upload-zone-corner tl" />
              <div className="upload-zone-corner tr" />
              <div className="upload-zone-corner bl" />
              <div className="upload-zone-corner br" />

              <div className="upload-icon-wrap">
                <UploadCloud className="upload-icon" />
              </div>

              <p className="upload-text">DROP FILE TO ANALYZE</p>
              <p className="upload-subtext">JPG · PNG · WEBP · MP4 · AVI · MKV · WEBM &nbsp;·&nbsp; MAX 50 MB</p>
              <p className="upload-prompt">CLICK OR DRAG A FILE HERE<span className="cursor" /></p>

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

          {/* ── LOADING STATE ───────────────────────────────── */}
          {loading && (
            <div className="loader-container" id="loader">
              <div className="cyber-spinner">
                <div className="cyber-spinner-ring" />
                <div className="cyber-spinner-ring" />
                <div className="cyber-spinner-ring" />
                <div className="cyber-spinner-core" />
              </div>

              <div className="scan-progress-track">
                <div className="scan-progress-fill" />
              </div>

              <p className="pulsing-text">{scanLine}</p>

              <div className="scan-log">
                <span>// LOADING CHECKPOINT: best_model.pth</span>
                <span>// FACE DETECTOR: MEDIAPIPE BLAZEFACE</span>
                {file?.type.includes('video') && (
                  <span>// VIDEO MODE: SAMPLING EVERY 8TH FRAME</span>
                )}
              </div>
            </div>
          )}

          {/* ── ERROR STATE ─────────────────────────────────── */}
          {error && !loading && (
            <div className="error-box" id="error-box">
              <h3><AlertTriangle size={14} strokeWidth={1.5} style={{ display: 'inline', marginRight: '6px' }} />ERROR</h3>
              <p className="detail-text" style={{ marginTop: '0.5rem' }}>{error}</p>
              <button
                className="reset-btn"
                id="try-again-btn"
                style={{ marginTop: '1.25rem' }}
                onClick={resetState}
              >
                <span>RETRY</span>
              </button>
            </div>
          )}

          {/* ── RESULTS VIEW ────────────────────────────────── */}
          {result && !loading && (
            <div className="results-view" id="results-view">

              {/* Side-by-side: original + Grad-CAM */}
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

              {/* ── VERDICT CARD ──────────────────────────── */}
              <div className={`verdict-box ${verdictClass}`} id="verdict-box">
                <div className="verdict-label" id="verdict-label">
                  {verdictIcon[verdictClass]}
                  &nbsp;{result.label}
                </div>

                <p className="detail-text">
                  FILE: <strong style={{ color: 'var(--fg)' }}>{result.filename}</strong>
                </p>

                {/* Confidence bar */}
                <div className="confidence-bar-bg">
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                  />
                </div>

                <div className="stats-row">
                  <span>
                    CONFIDENCE: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                  </span>
                  <span>
                    FAKE PROB: <strong>{result.fake_prob.toFixed(4)}</strong>
                  </span>
                </div>

                {result.label === 'UNCERTAIN' && (
                  <p className="detail-text uncertain-note">
                    ⚠ LOW CONFIDENCE (P=0.35–0.50). HUMAN REVIEW RECOMMENDED.
                  </p>
                )}
              </div>

              {/* Reset button */}
              <button className="reset-btn" id="reset-btn" onClick={resetState}>
                <RotateCcw size={14} strokeWidth={1.5} />
                <span>ANALYZE ANOTHER FILE</span>
              </button>

            </div>
          )}

        </div>{/* /panel-body */}

        {/* ── PANEL FOOTER ─────────────────────────────────── */}
        <div className="panel-footer">
          <div className="footer-stat">
            <span>STATUS:</span>
            <span className="footer-accent">■ READY</span>
          </div>
          <div className="footer-stat">
            <span>BACKEND:</span>
            <span className="footer-accent">localhost:8000</span>
          </div>
          <div className="footer-stat">
            <span>MODEL:</span>
            <span className="footer-accent">AUC 0.9391</span>
          </div>
          <div className="footer-stat">
            <span>DATASET:</span>
            <span className="footer-accent">DF40 · 40 METHODS</span>
          </div>
        </div>

      </div>{/* /cyber-panel */}

    </div>
  );
}

export default App;
