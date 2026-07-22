const WebSocket = require('ws');
const fs = require('fs');

const ws = new WebSocket('ws://localhost:5000/ws');

ws.on('open', () => {
  console.log('Connected to WebSocket');
});

let frames = 0;
ws.on('message', (data) => {
  const msg = JSON.parse(data);
  console.log(`Received: ${msg.type}`);
  if (msg.type === 'frame') {
    frames++;
    console.log(`Frame ${frames}: progress ${msg.progress}`);
  }
});

ws.on('close', () => {
  console.log('WebSocket closed');
  process.exit(0);
});

ws.on('error', (err) => {
  console.error('WebSocket error:', err);
});
