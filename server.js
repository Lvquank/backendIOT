const express = require('express');
const cors    = require('cors');
const db      = require('./database'); // Khởi tạo DB ngay khi start

const app  = express();
const PORT = 3000;

// ======== MIDDLEWARE ========
app.use(cors());
app.use(express.json());

// Log mỗi request
app.use((req, res, next) => {
  console.log(`[${new Date().toLocaleTimeString()}] ${req.method} ${req.path}`);
  next();
});

// ======== ROUTES ========
app.use('/api/auth',     require('./routes/auth'));
app.use('/api/users',    require('./routes/users'));
app.use('/api/vehicles', require('./routes/vehicles'));
app.use('/api/logs',     require('./routes/logs'));
app.use('/api/plate',    require('./routes/plate'));  // Webcam biển số

// Health check
app.get('/', (req, res) => {
  res.json({
    status:  'OK',
    message: 'ESP32 Parking API',
    version: '1.0.0',
    endpoints: [
      'POST /api/auth/verify',
      'GET  /api/users',
      'POST /api/users',
      'PUT  /api/users/:id',
      'DELETE /api/users/:id',
      'GET  /api/vehicles',
      'POST /api/vehicles',
      'DELETE /api/vehicles/:id',
      'GET  /api/logs',
      'GET  /api/logs/stats',
      'POST /api/plate/scan',
      'GET  /api/plate/latest',
      'GET  /api/plate/status',
    ]
  });
});

// ======== START SERVER ========
const os = require('os');

function getLocalIP() {
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return 'localhost';
}

app.listen(PORT, '0.0.0.0', () => {
  const ip = getLocalIP();
  console.log('\n==============================');
  console.log(' ESP32 Parking API - RUNNING');
  console.log('==============================');
  console.log(` Local:   http://localhost:${PORT}`);
  console.log(` Network: http://${ip}:${PORT}  ← ESP32 dùng IP này`);
  console.log('==============================');
  console.log(' ESP32 Arduino config:');
  console.log(`   #define API_HOST "${ip}"`);
  console.log(`   #define API_PORT ${PORT}`);
  console.log('==============================\n');
});
