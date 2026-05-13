const express = require('express');
const router = express.Router();
const db = require('../database');

let latestScan = null;

function normalizePlate(value) {
  return String(value || '').trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
}

function normalizeStatus(value) {
  return String(value || '').trim().toUpperCase() === 'OUT' ? 'OUT' : 'IN';
}

function findVehicleByPlate(plate) {
  const normalized = normalizePlate(plate);
  return db.prepare(`
    SELECT v.*, u.username, u.full_name
    FROM vehicles v
    JOIN users u ON u.id = v.user_id
    WHERE UPPER(REPLACE(REPLACE(REPLACE(v.plate_number, '-', ''), '.', ''), ' ', '')) = ?
  `).get(normalized);
}

function getLatestGrantedPlateLog(plate) {
  const normalized = normalizePlate(plate);
  return db.prepare(`
    SELECT *
    FROM access_logs
    WHERE granted = 1
      AND UPPER(REPLACE(REPLACE(REPLACE(COALESCE(plate_number, ''), '-', ''), '.', ''), ' ', '')) = ?
    ORDER BY timestamp DESC, id DESC
    LIMIT 1
  `).get(normalized);
}

function buildDecision(status, vehicle, plate) {
  if (!vehicle) {
    return {
      granted: false,
      inParking: false,
      message: 'Bien so khong co trong he thong',
      note: `Bien so ${plate} -> Khong tim thay trong he thong`,
    };
  }

  if (status === 'IN') {
    return {
      granted: true,
      inParking: true,
      message: `Tim thay: ${vehicle.full_name}`,
      note: `Bien so ${plate} -> ${vehicle.full_name} (${vehicle.username}) vao bai`,
    };
  }

  const latestLog = getLatestGrantedPlateLog(vehicle.plate_number);
  const inParking = latestLog && latestLog.status === 'IN';

  if (!inParking) {
    return {
      granted: false,
      inParking: false,
      message: 'Xe chua o trong bai, khong the quet OUT',
      note: `Bien so ${plate} -> Da dang ky nhung chua co trang thai IN hop le`,
    };
  }

  return {
    granted: true,
    inParking: false,
    message: `Cho phep ra: ${vehicle.full_name}`,
    note: `Bien so ${plate} -> ${vehicle.full_name} (${vehicle.username}) ra bai`,
  };
}

// POST /api/plate/scan
// Body: { "plate_number": "51A-12345", "status": "IN" | "OUT" }
router.post('/scan', (req, res) => {
  const { plate_number } = req.body;
  const scanStatus = normalizeStatus(req.body.status);
  const plate = normalizePlate(plate_number);

  if (!plate || plate.length < 3) {
    return res.status(400).json({ error: 'plate_number khong hop le' });
  }

  const vehicle = findVehicleByPlate(plate);
  const decision = buildDecision(scanStatus, vehicle, plate);

  if (decision.granted) {
    db.prepare(
      'INSERT INTO access_logs (user_id, username, plate_number, granted, status, note) VALUES (?, ?, ?, ?, ?, ?)'
    ).run(
      vehicle.user_id,
      vehicle.username,
      vehicle.plate_number,
      1,
      scanStatus,
      decision.note
    );
  }

  latestScan = {
    plate_number: plate,
    registered_plate_number: vehicle ? vehicle.plate_number : null,
    status: scanStatus,
    timestamp: new Date().toISOString(),
    matched: decision.granted,
    registered: !!vehicle,
    in_parking: decision.inParking,
    matched_user: vehicle ? vehicle.username : null,
    matched_full_name: vehicle ? vehicle.full_name : null,
    consumed: false,
  };

  console.log(
    `[PLATE] ${scanStatus}: ${plate} -> ${decision.granted ? 'OK' : 'DENIED'}`
  );

  return res.json({
    plate_number: plate,
    registered_plate_number: vehicle ? vehicle.plate_number : null,
    status: scanStatus,
    matched: decision.granted,
    registered: !!vehicle,
    in_parking: decision.inParking,
    user: vehicle
      ? { username: vehicle.username, full_name: vehicle.full_name }
      : null,
    message: decision.message,
  });
});

// GET /api/plate/latest
router.get('/latest', (req, res) => {
  if (!latestScan) {
    return res.json({ available: false });
  }

  const scan = { ...latestScan, available: true };
  latestScan.consumed = true;

  return res.json(scan);
});

// GET /api/plate/status
router.get('/status', (req, res) => {
  return res.json({
    has_pending: latestScan !== null && !latestScan.consumed,
    latest: latestScan,
  });
});

// DELETE /api/plate/clear
router.delete('/clear', (req, res) => {
  latestScan = null;
  return res.json({ message: 'Da xoa scan hien tai' });
});

module.exports = router;
