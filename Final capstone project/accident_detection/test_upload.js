const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');

const projectRoot = __dirname;
const pythonExec = path.join(projectRoot, 'venv', 'Scripts', 'python.exe');
const workerScript = path.join(projectRoot, 'backend', 'services', 'streaming', 'worker.py');

const args = [
  workerScript,
  '--task', 'collision',
  '--source', path.join(projectRoot, '..', 'Real_dataset_accident', 'real_videos', '-2UPLUV7JLg_00.mp4'),
  '--confidence_threshold', '0.85',
  '--export'
];

console.log('Spawning worker...', args);

const activeWorker = spawn(pythonExec, args, {
  cwd: projectRoot,
  env: { ...process.env, PYTHONPATH: projectRoot }
});

let lineBuffer = '';
let frameCount = 0;

activeWorker.stdout.on('data', (data) => {
  lineBuffer += data.toString();
  const lines = lineBuffer.split('\n');
  lineBuffer = lines.pop();

  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line);
      console.log(`Received:`, parsed);
      if (parsed.type === 'frame') {
        frameCount++;
        console.log(`Frame ${frameCount}: progress ${parsed.progress}, fps ${parsed.fps}`);
      }
    } catch (err) {
      console.log(`[Python Worker Output]: ${line.substring(0, 100)}...`);
    }
  }
});

activeWorker.stderr.on('data', (data) => {
  console.error(`[Python Worker Error]: ${data}`);
});

activeWorker.on('close', (code) => {
  console.log(`[!] Worker child process exited with code ${code}. Total frames received: ${frameCount}`);
});
