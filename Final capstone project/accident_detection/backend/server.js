import express from 'express';
import { WebSocketServer } from 'ws';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import cors from 'cors';
import multer from 'multer';
import sqlite3 from 'sqlite3';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const NEW_DATASET_ROOT = 'D:/Subjects_Languages/Languages/VED-DEVANAND-DHANOKAR-g37-ai-ml/Final capstone project/Real_dataset_accident';

const app = express();
const PORT = process.env.PORT || 8000;

app.use(cors());
app.use(express.json());

// Serve static assets: snapshots, logs, outputs
app.use('/logs', express.static(path.join(projectRoot, 'logs')));
app.use('/data/raw', express.static(path.join(projectRoot, 'data', 'raw')));

// Configure Multer for video uploads
const uploadDir = path.join(projectRoot, 'logs', 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `upload_${Date.now()}${ext}`);
  }
});
const upload = multer({ storage });

// Active Child Process state
let activeWorker = null;
let activeTask = null;
let activeSource = null;
let activeClients = new Set();

// ── WebSocket Server Setup ──────────────────────────────
const wss = new WebSocketServer({ noServer: true });

wss.on('connection', (ws) => {
  activeClients.add(ws);
  console.log(`[+] WS client connected. Total clients: ${activeClients.size}`);

  // Send initial status
  ws.send(JSON.stringify({
    type: 'status',
    task: activeTask,
    source: activeSource,
    status: activeWorker ? 'running' : 'idle'
  }));

  ws.on('close', () => {
    activeClients.delete(ws);
    console.log(`[-] WS client disconnected. Total clients: ${activeClients.size}`);
  });
});

function broadcast(data) {
  const payload = typeof data === 'string' ? data : JSON.stringify(data);
  for (const client of activeClients) {
    if (client.readyState === 1) { // OPEN
      client.send(payload);
    }
  }
}

// ── Child Process Spawner ───────────────────────────────
function startWorker(task, source, confidence = 0.85, exportVideo = true) {
  if (activeWorker) {
    stopWorker();
  }

  activeTask = task;
  activeSource = source;

  const pythonExec = path.join(projectRoot, 'venv', 'Scripts', 'python.exe');
  const workerScript = path.join(__dirname, 'services', 'streaming', 'worker.py');

  const args = [
    workerScript,
    '--task', task,
    '--source', String(source),
    '--confidence_threshold', String(confidence)
  ];

  if (exportVideo) {
    args.push('--export');
  }

  console.log(`[+] Spawning child: ${pythonExec} ${args.join(' ')}`);

  activeWorker = spawn(pythonExec, args, {
    cwd: projectRoot,
    env: { ...process.env, PYTHONPATH: projectRoot }
  });

  let lineBuffer = '';

  activeWorker.stdout.on('data', (data) => {
    lineBuffer += data.toString();
    const lines = lineBuffer.split('\n');
    lineBuffer = lines.pop(); // Keep incomplete line

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const parsed = JSON.parse(line);
        // Forward all frames and metrics directly via WebSockets
        broadcast(parsed);
      } catch (err) {
        // Fallback for standard prints or debugging
        console.log(`[Python Worker Output]: ${line}`);
      }
    }
  });

  activeWorker.stderr.on('data', (data) => {
    console.error(`[Python Worker Error]: ${data}`);
  });

  activeWorker.on('close', (code) => {
    console.log(`[!] Worker child process exited with code ${code}`);
    broadcast({ type: 'exit', code });
    activeWorker = null;
    activeTask = null;
    activeSource = null;
  });
}

function stopWorker() {
  if (activeWorker) {
    console.log('[*] Terminating python worker process...');
    activeWorker.kill('SIGINT');
    activeWorker = null;
    activeTask = null;
    activeSource = null;
    return true;
  }
  return false;
}

// ── REST Endpoints ─────────────────────────────────────

// Upload file endpoint
app.post('/api/upload-video', upload.single('video'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file provided' });
  }

  const { task, confidence } = req.body;
  const filePath = req.file.path;

  // Start processing task
  startWorker(task || 'collision', filePath, confidence || 0.85, true);

  res.json({
    message: 'Upload successful. Processing started.',
    filename: req.file.filename,
    path: filePath
  });
});

// Start webcam endpoint
app.post('/api/start-webcam', (req, res) => {
  const { task, index, confidence } = req.body;
  const sourceIndex = index !== undefined ? index : 0;

  startWorker(task || 'collision', sourceIndex, confidence || 0.85, false);

  res.json({
    message: 'Webcam detection started.',
    source: sourceIndex
  });
});

// Start RTSP or custom video by path/demo endpoint
app.post('/api/start-source', (req, res) => {
  const { task, source, confidence } = req.body;
  if (!source) {
    return res.status(400).json({ error: 'No source path provided' });
  }

  // Resolve demo video paths if needed
  let finalSource = source;
  if (source.startsWith('demo:')) {
    const filename = source.replace('demo:', '');
    finalSource = path.join(NEW_DATASET_ROOT, 'real_videos', filename);
    if (!fs.existsSync(finalSource)) {
      finalSource = path.join(NEW_DATASET_ROOT, 'synthetic_videos', filename);
    }
  }

  startWorker(task || 'collision', finalSource, confidence || 0.85, true);

  res.json({
    message: `Processing started for source: ${source}`,
    resolvedPath: finalSource
  });
});

// Stop active task
app.post('/api/stop', (req, res) => {
  const stopped = stopWorker();
  res.json({ stopped, status: 'idle' });
});

// Status check
app.get('/api/status', (req, res) => {
  res.json({
    status: activeWorker ? 'running' : 'idle',
    task: activeTask,
    source: activeSource
  });
});

// Get demo video lists
app.get('/api/demo-videos', (req, res) => {
  const realVideosDir = path.join(NEW_DATASET_ROOT, 'real_videos');
  const syntheticVideosDir = path.join(NEW_DATASET_ROOT, 'synthetic_videos');
  let files = [];

  const addFilesFromDir = (dir) => {
    if (fs.existsSync(dir)) {
      const dirFiles = fs.readdirSync(dir).filter(f => f.endsWith('.mp4') || f.endsWith('.avi'));
      files = files.concat(dirFiles);
    }
  };

  addFilesFromDir(realVideosDir);
  addFilesFromDir(syntheticVideosDir);

  // Return unique filenames
  res.json({ videos: [...new Set(files)] });
});

// Query recent collision incidents from SQLite
app.get('/api/incidents', (req, res) => {
  const dbPath = path.join(projectRoot, 'logs', 'incidents.db');
  
  if (!fs.existsSync(dbPath)) {
    return res.json({ incidents: [] });
  }

  const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READONLY, (err) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }
  });

  db.all('SELECT * FROM incidents ORDER BY id DESC LIMIT 50', [], (err, rows) => {
    db.close();
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    res.json({ incidents: rows });
  });
});

// Clear collision database log and associated snapshots
app.post('/api/clear-incidents', (req, res) => {
  const dbPath = path.join(projectRoot, 'logs', 'incidents.db');
  const snapshotsDir = path.join(projectRoot, 'logs', 'snapshots');

  const clearSnapshots = () => {
    if (fs.existsSync(snapshotsDir)) {
      try {
        const files = fs.readdirSync(snapshotsDir);
        for (const file of files) {
          const filePath = path.join(snapshotsDir, file);
          if (fs.statSync(filePath).isFile()) {
            fs.unlinkSync(filePath);
          }
        }
      } catch (err) {
        console.error(`Error deleting snapshots: ${err.message}`);
      }
    }
  };

  if (!fs.existsSync(dbPath)) {
    clearSnapshots();
    return res.json({ success: true, message: 'No incidents database to clear' });
  }

  const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READWRITE, (err) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }
  });

  db.run('DELETE FROM incidents', [], function(err) {
    db.close();
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    clearSnapshots();
    res.json({ success: true, message: 'All incidents and snapshots cleared successfully' });
  });
});

// Query helmet violations from CSV log file
app.get('/api/violations', (req, res) => {
  const csvPath = path.join(projectRoot, 'logs', 'helmet_violations.csv');

  if (!fs.existsSync(csvPath)) {
    return res.json({ violations: [] });
  }

  fs.readFile(csvPath, 'utf8', (err, data) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }

    const lines = data.trim().split('\n');
    if (lines.length <= 1) {
      return res.json({ violations: [] });
    }

    // Parse header and rows
    const headers = lines[0].split(',').map(h => h.replace(/"/g, '').trim());
    const violations = [];

    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',').map(c => c.replace(/"/g, '').trim());
      if (cols.length < headers.length) continue;

      const obj = {};
      headers.forEach((h, index) => {
        obj[h] = cols[index];
      });
      violations.push(obj);
    }

    // Return in reverse chronological order
    res.json({ violations: violations.reverse() });
  });
});

// Get processed export outputs
app.get('/api/outputs', (req, res) => {
  const outputsDir = path.join(projectRoot, 'logs', 'outputs');
  if (!fs.existsSync(outputsDir)) {
    return res.json({ outputs: [] });
  }

  fs.readdir(outputsDir, (err, files) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }

    const mp4s = files.filter(f => f.endsWith('.mp4')).map(f => {
      const stats = fs.statSync(path.join(outputsDir, f));
      return {
        filename: f,
        sizeMb: (stats.size / (1024 * 1024)).toFixed(2),
        createdAt: stats.mtime
      };
    });

    res.json({ outputs: mp4s.sort((a, b) => b.createdAt - a.createdAt) });
  });
});

// ── Startup & Server Listen ─────────────────────────────
const server = app.listen(PORT, () => {
  console.log(`[v] Express API Server listening on port ${PORT}`);
});

// Attach WebSocket Server to the same HTTP server
server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});
