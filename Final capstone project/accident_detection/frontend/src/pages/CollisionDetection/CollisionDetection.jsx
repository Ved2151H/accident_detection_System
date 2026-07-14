import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Upload, RefreshCw, AlertOctagon, MapPin, Eye, Info } from 'lucide-react';

function CollisionDetection({ theme }) {
  const [sourceType, setSourceType] = useState('demo');
  const [demoVideos, setDemoVideos] = useState([]);
  const [selectedDemo, setSelectedDemo] = useState('');
  const [webcamIndex, setWebcamIndex] = useState(0);
  const [rtspUrl, setRtspUrl] = useState('rtsp://192.168.1.100:554/stream');
  const [confidence, setConfidence] = useState(0.85);
  
  // Status states
  const [status, setStatus] = useState('idle'); // idle, running
  const [alertState, setAlertState] = useState('Normal'); // Normal, Warning, Confirmed Accident
  const [yoloConf, setYoloConf] = useState(0);
  const [lstmProb, setLstmProb] = useState(0);
  const [progress, setProgress] = useState(0);
  const [fps, setFps] = useState(0);
  const [features, setFeatures] = useState({});

  // Location telemetry
  const [location, setLocation] = useState({ lat: 18.5204, lon: 73.8567, city_name: 'Pune (West)', digipin: '4FP-492-CMTF' });
  const [incidentData, setIncidentData] = useState(null);

  // Live frame display
  const [currentFrame, setCurrentFrame] = useState(null);
  
  // Historical incidents list
  const [incidents, setIncidents] = useState([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);

  // Video upload ref
  const fileInputRef = useRef(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  // Map reference
  const mapRef = useRef(null);
  const markerRef = useRef(null);

  // WebSocket reference
  const wsRef = useRef(null);

  // Fetch demo videos and recent incidents on mount
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

    fetchIncidents();
  }, []);

  const fetchIncidents = () => {
    fetch('/api/incidents')
      .then(res => res.json())
      .then(data => setIncidents(data.incidents || []))
      .catch(err => console.error('Error fetching incidents:', err));
  };

  // Leaflet Map Initialization and updates
  useEffect(() => {
    if (!window.L) return;

    const lat = incidentData ? incidentData.lat : location.lat;
    const lon = incidentData ? incidentData.lon : location.lon;
    const label = incidentData ? `🚨 CRASH DETECTED\nDigiPIN: ${incidentData.digipin}` : `📍 Camera Location\nDigiPIN: ${location.digipin}`;

    if (!mapRef.current) {
      // Create map
      mapRef.current = window.L.map('collision-map').setView([lat, lon], 12);
      
      window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
      }).addTo(mapRef.current);

      markerRef.current = window.L.marker([lat, lon]).addTo(mapRef.current);
      markerRef.current.bindPopup(label.replace(/\n/g, '<br>')).openPopup();
    } else {
      // Update map center and marker
      mapRef.current.setView([lat, lon], 13);
      markerRef.current.setLatLng([lat, lon]);
      markerRef.current.bindPopup(label.replace(/\n/g, '<br>')).openPopup();
    }
  }, [location, incidentData]);

  // Handle Websocket connections
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
          setLocation(msg.location);
          setIncidentData(null);
        } else if (msg.type === 'frame') {
          setCurrentFrame(msg.frame);
          setAlertState(msg.alert_state);
          setYoloConf(msg.raw_prob);
          setLstmProb(msg.calibrated_prob);
          setProgress(msg.progress);
          setFps(msg.fps);
          setFeatures(msg.features || {});
        } else if (msg.type === 'incident') {
          setAlertState('Confirmed Accident');
          setIncidentData(msg);
          // Auto refresh logs
          fetchIncidents();
        } else if (msg.type === 'exit') {
          setStatus('idle');
          setCurrentFrame(null);
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WS connection closed.');
    };
  };

  const handleStart = async () => {
    connectWebSocket();
    setIncidentData(null);

    let body = { task: 'collision', confidence };
    let url = '/api/start-source';

    if (sourceType === 'webcam') {
      body.index = webcamIndex;
      url = '/api/start-webcam';
    } else if (sourceType === 'demo') {
      body.source = `demo:${selectedDemo}`;
    } else if (sourceType === 'rtsp') {
      body.source = rtspUrl;
    } else if (sourceType === 'upload') {
      if (!uploadedFile) {
        alert('Please choose a file to upload first.');
        return;
      }
      // Form data upload
      const formData = new FormData();
      formData.append('video', uploadedFile);
      formData.append('task', 'collision');
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
      console.log('Task started:', data);
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
      setIncidentData(null);
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleClearLogs = () => {
    if (!window.confirm('Are you sure you want to clear all incidents and snapshots?')) {
      return;
    }
    fetch('/api/clear-incidents', { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setIncidents([]);
          setSelectedSnapshot(null);
          setIncidentData(null);
        } else {
          alert('Failed to clear logs: ' + (data.error || 'Unknown error'));
        }
      })
      .catch(err => {
        console.error('Error clearing logs:', err);
        alert('Error clearing logs');
      });
  };

  const handleFileUploadChange = (e) => {
    if (e.target.files.length > 0) {
      setUploadedFile(e.target.files[0]);
    }
  };

  return (
    <div className="grid-container">
      {/* Left panel - Video & Live feedback */}
      <div>
        {incidentData && (
          <div className="alert-banner-red">
            <h3>🚨 COLLISION DETECTED & SYSTEM FROZEN</h3>
            <p>
              An active vehicle accident threat was detected at <strong>{new Date(incidentData.timestamp).toLocaleString()}</strong>.
              The camera feed has been frozen at the anomaly timestamp. Automatic rescue dispatch localizing.
            </p>
          </div>
        )}

        {!incidentData && status === 'running' && (
          <div className={`banner-green ${alertState === 'Warning' ? 'alert-banner-red' : ''}`}>
            <h4>🛰️ ACTIVE THREAT SCANNING</h4>
            <p>
              Monitoring camera feed index. Alert state: <strong>{alertState}</strong>.
            </p>
          </div>
        )}

        {status === 'idle' && (
          <div className="banner-green" style={{ background: 'rgba(15, 23, 42, 0.05)', borderColor: 'var(--border-color)' }}>
            <h4>🛰️ STANDBY / AWAITING SIGNAL</h4>
            <p>Inference core initialized. Configure options and start threat scanning.</p>
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
                  <p style={{ fontWeight: 'bold', fontSize: '1rem', color: 'var(--text-primary)' }}>Threat Monitoring Standby</p>
                  <p style={{ fontSize: '0.8rem', marginTop: '4px' }}>Select an input stream and click start in the settings panel to begin.</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Incidents history */}
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>📋 Recent Incidents Database Log</span>
            <button className="btn-clear-logs" onClick={handleClearLogs}>
              🗑️ Clear Logs
            </button>
          </div>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Source</th>
                  <th>YOLO Conf</th>
                  <th>LSTM Conf</th>
                  <th>Snapshot</th>
                  <th>Geoloc</th>
                </tr>
              </thead>
              <tbody>
                {incidents.length === 0 ? (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>No accident incidents recorded.</td>
                  </tr>
                ) : (
                  incidents.map((inc) => (
                    <tr key={inc.id}>
                      <td>{new Date(inc.timestamp).toLocaleString()}</td>
                      <td>{inc.source}</td>
                      <td style={{ color: 'var(--danger-color)' }}>{(inc.yolo_conf * 100).toFixed(1)}%</td>
                      <td style={{ color: 'var(--danger-color)' }}>{(inc.lstm_prob * 100).toFixed(1)}%</td>
                      <td>
                        {inc.snapshot && (
                          <img
                            src={`/${inc.snapshot}`}
                            alt="Snapshot"
                            className="snapshot-thumb"
                            onClick={() => setSelectedSnapshot(`/${inc.snapshot}`)}
                          />
                        )}
                      </td>
                      <td>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '4px 8px', fontSize: '0.75rem', width: 'auto' }}
                          onClick={() => {
                            setIncidentData({
                              lat: inc.latitude,
                              lon: inc.longitude,
                              digipin: inc.digipin,
                              timestamp: inc.timestamp
                            });
                          }}
                        >
                          <MapPin size={12} /> View Map
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Right panel - Controls & Telemetry */}
      <div>
        <div className="card">
          <div className="card-header">⚙️ Accident Monitor Settings</div>

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
              <option value="rtsp">CCTV RTSP Network Stream</option>
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

          {sourceType === 'rtsp' && (
            <div className="form-group">
              <label className="form-label">RTSP Connection URL</label>
              <input
                type="text"
                className="form-input"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
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

          <div className="form-group">
            <label className="form-label">LSTM Calibration Threshold: {confidence}</label>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.01"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              style={{ width: '100%' }}
              disabled={status === 'running'}
            />
          </div>

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

        {/* Map panel */}
        <div className="card">
          <div className="card-header">📍 Camera Geolocation (Leaflet)</div>
          <div id="collision-map" className="map-container" />
          <div className="telemetry-grid">
            <div className="telemetry-card">
              <span className="telemetry-label">Assigned Location</span>
              <div className="telemetry-value">📍 {incidentData ? 'Accident Spot' : location.city_name}</div>
            </div>
            <div className="telemetry-card">
              <span className="telemetry-label">India Post DIGIPIN</span>
              <div className="telemetry-value" style={{ color: 'var(--accent-secondary)' }}>
                {incidentData ? incidentData.digipin : location.digipin}
              </div>
            </div>
          </div>
          <div className="telemetry-card" style={{ marginTop: '12px' }}>
            <span className="telemetry-label">Coordinates Info</span>
            <div className="telemetry-details">
              <strong>LATITUDE:</strong> {incidentData ? incidentData.lat : location.lat}° N<br />
              <strong>LONGITUDE:</strong> {incidentData ? incidentData.lon : location.lon}° E
            </div>
          </div>
        </div>

        {/* Scan Telemetry Panel */}
        <div className="card">
          <div className="card-header">📊 Scan Telemetry Metrics</div>
          <div className="telemetry-grid">
            <div className="telemetry-card">
              <span className="telemetry-label">YOLO Probability</span>
              <div className="telemetry-value" style={{ color: yoloConf > confidence ? 'var(--danger-color)' : 'var(--success-color)' }}>
                {(yoloConf * 100).toFixed(1)}%
              </div>
            </div>
            <div className="telemetry-card">
              <span className="telemetry-label">LSTM Threat Score</span>
              <div className="telemetry-value" style={{ color: lstmProb > confidence ? 'var(--danger-color)' : 'var(--success-color)' }}>
                {(lstmProb * 100).toFixed(1)}%
              </div>
            </div>
            <div className="telemetry-card" style={{ gridColumn: 'span 2', marginTop: '4px' }}>
              <span className="telemetry-label">Processing Performance</span>
              <div className="telemetry-details">
                <strong>Current Frame:</strong> {progress > 0 ? `${progress.toFixed(1)}%` : '0%'} ({fps.toFixed(1)} FPS)
                <div className="progress-bar-container">
                  <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
                </div>
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
            <img src={selectedSnapshot} alt="Accident Snapshot" className="modal-img" />
          </div>
        </div>
      )}
    </div>
  );
}

export default CollisionDetection;
