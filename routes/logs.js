const express = require('express');
const router = express.Router();
const db = require('../database');

const webcamLogWhere = "COALESCE(plate_number, '') <> ''";

// GET /api/logs - Lich su webcam quet bien so
router.get('/', (req, res) => {
  const limit = parseInt(req.query.limit) || 100;
  const logs = db.prepare(`
    SELECT * FROM access_logs
    WHERE ${webcamLogWhere}
    ORDER BY timestamp DESC
    LIMIT ?
  `).all(limit);
  res.json(logs);
});

// GET /api/logs/user/:username - Lich su webcam cua 1 user
router.get('/user/:username', (req, res) => {
  const logs = db.prepare(`
    SELECT * FROM access_logs
    WHERE username = ?
      AND ${webcamLogWhere}
    ORDER BY timestamp DESC
    LIMIT 50
  `).all(req.params.username);
  res.json(logs);
});

// GET /api/logs/stats - Thong ke nhanh lich su webcam
router.get('/stats', (req, res) => {
  const total = db.prepare(`
    SELECT COUNT(*) as cnt FROM access_logs
    WHERE ${webcamLogWhere}
  `).get();
  const granted = db.prepare(`
    SELECT COUNT(*) as cnt FROM access_logs
    WHERE granted = 1 AND ${webcamLogWhere}
  `).get();
  const denied = db.prepare(`
    SELECT COUNT(*) as cnt FROM access_logs
    WHERE granted = 0 AND ${webcamLogWhere}
  `).get();
  const inCount = db.prepare(`
    SELECT COUNT(*) as cnt FROM access_logs
    WHERE status = 'IN' AND ${webcamLogWhere}
  `).get();
  const outCount = db.prepare(`
    SELECT COUNT(*) as cnt FROM access_logs
    WHERE status = 'OUT' AND ${webcamLogWhere}
  `).get();
  const today = db.prepare(`
    SELECT COUNT(*) as cnt FROM access_logs
    WHERE date(timestamp) = date('now','localtime')
      AND ${webcamLogWhere}
  `).get();

  res.json({
    total: total.cnt,
    granted: granted.cnt,
    denied: denied.cnt,
    in: inCount.cnt,
    out: outCount.cnt,
    today: today.cnt,
  });
});

module.exports = router;
