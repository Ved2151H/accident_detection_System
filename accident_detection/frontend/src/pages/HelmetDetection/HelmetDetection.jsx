import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Upload, Eye, ShieldAlert, Award, FileText } from 'lucide-react';

function HelmetDetection({ theme }) {
  const [sourceType, setSourceType] = useState('demo');
  const [demoVideos, setDemoVideos] = useState([]);
  const [selectedDemo, setSelectedDemo] = useState('');
  const [webcamIndex, setWebcamIndex] = useState(0);
  const [confidence, setConfidence] = useState(0.85);
  
  // Status states
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [fps, setFps] = useState(0);
  const [indicators, setIndicators] = useState({ bike: false, rider: false, helmet: false });
  const [stats, setStats] = useState({ total_bikes: 0, helmet_users: 0, non_helmet_riders: 0, compliance_pct: 0 });

  // Live frame display
  const [currentFrame, setCurrentFrame] = useState(null);
  
  // Violation list
  const [violations, setViolations] = useState([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);

  // Video upload ref
  const fileInputRef = useRef(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  // WebSocket reference
  const wsRef = useRef(null);

  useEffect(() => {
    fetch('/api/demo-videos')
      .then(res => res.json())
      .then(data => {
        setDemoVideos(data.videos || []);
        if (data.videos && data.videos.length > 0) {
          setSelectedDemo(data.videos[0]);
        }
      })
      .catch(err => console.error('Error fetching demo videos:', err));

    fetchViolations();
  }, []);

  const fetchViolations = () => {
    fetch('/api/violations')
      .then(res => res.json())
      .then(data => setViolations(data.violations || []))
      .catch(err => console.error('Error fetching violations:', err));
  };

  const connectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${window.location.host}/ws`;
    console.log(`Connecting to WS: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'start') {
          setStatus('running');
        } else if (msg.type === 'frame') {
          setCurrentFrame(msg.frame);
          setProgress(msg.progress);
          setFps(msg.fps);
          setIndicators({
            bike: msg.bike_present,
            rider: msg.rider_present,
            helmet: msg.helmet_status
          });
          if (msg.stats) {
            setStats(msg.stats);
          }
        } else if (msg.type === 'violation') {
          // Refresh list on new violation
          fetchViolations();
        } else if (msg.type === 'exit') {
          setStatus('idle');
          setCurrentFrame(null);
          setIndicators({ bike: false, rider: false, helmet: false });
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WS closed.');
    };
  };

  const handleStart = async () => {
    connectWebSocket();

    let body = { task: 'helmet', confidence };
    let url = '/api/start-source';

    if (sourceType === 'webcam') {
      body.index = webcamIndex;
      url = '/api/start-webcam';
    } else if (sourceType === 'demo') {
      body.source = `demo:${selectedDemo}`;
    } else if (sourceType === 'upload') {
      if (!uploadedFile) {
        alert('Please choose a file to upload first.');
        return;
      }
      const formData = new FormData();
      formData.append('video', uploadedFile);
      formData.append('task', 'helmet');
      formData.append('confidence', confidence);
      
      setStatus('running');
      fetch('/api/upload-video', {
        method: 'POST',
        body: formData
      })
        .then(res => res.json())
        .then(data => console.log('Upload started:', data))
        .catch(err => {
          console.error(err);
          setStatus('idle');
        });
      return;
    }

    setStatus('running');
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      console.log('Helmet task started:', data);
    } catch (err) {
      console.error(err);
      setStatus('idle');
    }
  };

  const handleStop = async () => {
    try {
      await fetch('/api/stop', { method: 'POST' });
      setStatus('idle');
      setCurrentFrame(null);
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileUploadChange = (e) => {
    if (e.target.files.length > 0) {
      setUploadedFile(e.target.files[0]);
    }
  };

  // Determine indicator colors
  const getIndicatorStyle = (active, label) => {
    const isDark = theme === 'cyberpunk' || theme === 'slate';
    if (active) {
      return {
        background: isDark ? 'rgba(0, 255, 102, 0.08)' : 'rgba(22, 163, 74, 0.08)',
        border: isDark ? '1px solid rgba(0, 255, 102, 0.3)' : '1px solid rgba(22, 163, 74, 0.3)',
        color: isDark ? '#00ff66' : '#16a34a'
      };
    } else {
      return {
        background: isDark ? 'rgba(255, 51, 102, 0.08)' : 'rgba(220, 38, 38, 0.08)',
        border: isDark ? '1px solid rgba(255, 51, 102, 0.3)' : '1px solid rgba(220, 38, 38, 0.3)',
        color: isDark ? '#ff3366' : '#dc2626'
      };
    }
  };

  const complianceColor = stats.compliance_pct >= 80 
    ? 'var(--success-color)' 
    : (stats.compliance_pct >= 50 ? '#eab308' : 'var(--danger-color)');

  return (
    <div className="grid-container">
      {/* Left panel - Video & Live feedback */}
      <div>
        {status === 'running' && (
          <div className="banner-green">
            <h4>🛰️ COMPLIANCE SCAN ACTIVE</h4>
            <p>Scanning motorcycles for rider and helmet compliance rules.</p>
          </div>
        )}

        {status === 'idle' && (
          <div className="banner-green" style={{ background: 'rgba(15, 23, 42, 0.05)', borderColor: 'var(--border-color)' }}>
            <h4>🛰️ STANDBY / AWAITING SIGNAL</h4>
            <p>Helmet tracker ready. Configure input stream and start monitoring.</p>
          </div>
        )}

        <div className="card">
          <div className="card-header">🛡️ Live Stream Analysis</div>
          <div className="video-display-wrapper">
            {currentFrame ? (
              <img src={currentFrame} alt="Annotated feed" className="video-frame" />
            ) : (
              <div className="video-overlay-standby">
                <Eye size={48} />
                <div>
                  <p style={{ fontWeight: 'bold', fontSize: '1rem', color: 'var(--text-primary)' }}>Compliance Scan Standby</p>
                  <p style={{ fontSize: '0.8rem', marginTop: '4px' }}>Select an input stream and click start in the settings panel to begin.</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Live Indicators Row */}
        <div className="card">
          <div className="card-header">🛰️ Live Status Indicators</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            <div style={{ ...getIndicatorStyle(indicators.bike), borderRadius: '12px', padding: '16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
              <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>🏍️ Bike Present</span>
              <div style={{ fontSize: '1.6rem', fontWeight: 'bold', marginTop: '4px' }}>{indicators.bike ? 'YES' : 'NO'}</div>
            </div>
            <div style={{ ...getIndicatorStyle(indicators.rider), borderRadius: '12px', padding: '16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
              <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>👤 Rider Present</span>
              <div style={{ fontSize: '1.6rem', fontWeight: 'bold', marginTop: '4px' }}>{indicators.rider ? 'YES' : 'NO'}</div>
            </div>
            <div style={{ ...getIndicatorStyle(indicators.helmet), borderRadius: '12px', padding: '16px', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
              <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>🪖 Helmet Status</span>
              <div style={{ fontSize: '1.6rem', fontWeight: 'bold', marginTop: '4px' }}>{indicators.helmet ? 'OK' : 'NO'}</div>
            </div>
          </div>
        </div>

        {/* Violation Log Table */}
        <div className="card">
          <div className="card-header">📋 Recent Violation Logs</div>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Motorcycle ID</th>
                  <th>Violation Type</th>
                  <th>Evidence Snapshot</th>
                  <th>License Plate</th>
                  <th>Challan Status</th>
                </tr>
              </thead>
              <tbody>
                {violations.length === 0 ? (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>No violations logged yet.</td>
                  </tr>
                ) : (
                  violations.map((v, index) => (
                    <tr key={index}>
                      <td>{v.Timestamp}</td>
                      <td>Bike #{v['Motorcycle Track ID']}</td>
                      <td style={{ color: 'var(--danger-color)', fontWeight: 'bold' }}>{v['Violation Type']}</td>
                      <td>
                        {v['Snapshot Path'] && (
                          <img
                            src={`/${v['Snapshot Path']}`}
                            alt="Evidence Snapshot"
                            className="snapshot-thumb"
                            onClick={() => setSelectedSnapshot(`/${v['Snapshot Path']}`)}
                          />
                        )}
                      </td>
                      <td><span style={{ fontFamily: 'var(--font-mono)', padding: '2px 6px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}>{v['License Plate (ANPR Stub)']}</span></td>
                      <td style={{ color: 'rgba(255, 165, 0, 0.95)' }}>{v['Challan Status (Stub)']}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Right panel - Controls & Stats */}
      <div>
        <div className="card">
          <div className="card-header">⚙️ Helmet Monitor Settings</div>

          <div className="form-group">
            <label className="form-label">Input Source</label>
            <select
              className="form-select"
              value={sourceType}
              onChange={(e) => setSourceType(e.target.value)}
              disabled={status === 'running'}
            >
              <option value="demo">High-Res Demo Footage</option>
              <option value="webcam">Laptop / USB Webcam</option>
              <option value="upload">Upload Custom Video</option>
            </select>
          </div>

          {sourceType === 'demo' && (
            <div className="form-group">
              <label className="form-label">Select Demo Video Clip</label>
              <select
                className="form-select"
                value={selectedDemo}
                onChange={(e) => setSelectedDemo(e.target.value)}
                disabled={status === 'running'}
              >
                {demoVideos.map(vid => (
                  <option key={vid} value={vid}>{vid}</option>
                ))}
              </select>
            </div>
          )}

          {sourceType === 'webcam' && (
            <div className="form-group">
              <label className="form-label">Webcam Index</label>
              <input
                type="number"
                className="form-input"
                min="0"
                max="10"
                value={webcamIndex}
                onChange={(e) => setWebcamIndex(parseInt(e.target.value))}
                disabled={status === 'running'}
              />
            </div>
          )}

          {sourceType === 'upload' && (
            <div className="form-group">
              <label className="form-label">Upload Video File</label>
              <div 
                style={{
                  border: '2px dashed var(--border-color)',
                  borderRadius: '8px',
                  padding: '20px',
                  textAlign: 'center',
                  cursor: 'pointer'
                }}
                onClick={() => fileInputRef.current.click()}
              >
                <Upload size={24} style={{ margin: '0 auto 8px auto', color: 'var(--text-secondary)' }} />
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {uploadedFile ? uploadedFile.name : 'Click to select and upload video'}
                </span>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="video/mp4,video/avi"
                  onChange={handleFileUploadChange}
                  disabled={status === 'running'}
                />
              </div>
            </div>
          )}

          <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
            {status === 'idle' ? (
              <button className="btn btn-primary" onClick={handleStart}>
                <Play size={16} /> Start System
              </button>
            ) : (
              <button className="btn btn-danger" onClick={handleStop}>
                <Square size={16} /> Stop System
              </button>
            )}
          </div>
        </div>

        {/* Stats card */}
        <div className="card">
          <div className="card-header">📊 Helmet Statistics Panel</div>
          
          <div className="telemetry-card" style={{ marginBottom: '12px' }}>
            <span className="telemetry-label">Total Bikes Detected</span>
            <div className="metric-value">{stats.total_bikes}</div>
          </div>
          
          <div className="telemetry-card" style={{ marginBottom: '12px' }}>
            <span className="telemetry-label">Helmet Users (Compliant)</span>
            <div className="metric-value" style={{ color: 'var(--success-color)' }}>{stats.helmet_users}</div>
          </div>
          
          <div className="telemetry-card" style={{ marginBottom: '12px' }}>
            <span className="telemetry-label">Non-Helmet Riders (Violations)</span>
            <div className="metric-value" style={{ color: 'var(--danger-color)' }}>{stats.non_helmet_riders}</div>
          </div>
          
          <div className="telemetry-card" style={{ position: 'relative' }}>
            <span className="telemetry-label">Helmet Compliance Rate</span>
            <div className="metric-value" style={{ color: complianceColor }}>
              {stats.compliance_pct.toFixed(1)}%
            </div>
            <div className="progress-bar-container">
              <div 
                className="progress-bar-fill" 
                style={{ 
                  width: `${stats.compliance_pct}%`,
                  background: complianceColor,
                  boxShadow: `0 0 8px ${complianceColor}`
                }} 
              />
            </div>
          </div>
        </div>

        {/* Stream Progress Panel */}
        <div className="card">
          <div className="card-header">📊 Stream Performance</div>
          <div className="telemetry-card">
            <span className="telemetry-label">Frame Progress</span>
            <div className="telemetry-details" style={{ marginTop: '4px' }}>
              <strong>Progress:</strong> {progress > 0 ? `${progress.toFixed(1)}%` : '0%'} ({fps.toFixed(1)} FPS)
              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Snapshot Preview Modal */}
      {selectedSnapshot && (
        <div className="modal-overlay" onClick={() => setSelectedSnapshot(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedSnapshot(null)}>×</button>
            <img src={selectedSnapshot} alt="Evidence Snapshot" className="modal-img" />
          </div>
        </div>
      )}
    </div>
  );
}

export default HelmetDetection;
