const express = require('express');
const router  = express.Router();
const db      = require('../database');

// GET /api/vehicles - Tất cả biển số
router.get('/', (req, res) => {
  const vehicles = db.prepare(`
    SELECT v.*, u.username, u.full_name
    FROM vehicles v
    JOIN users u ON u.id = v.user_id
    ORDER BY v.created_at DESC
  `).all();
  res.json(vehicles);
});

// GET /api/vehicles/user/:userId - Biển số của 1 user
router.get('/user/:userId', (req, res) => {
  const vehicles = db.prepare(
    'SELECT * FROM vehicles WHERE user_id = ? ORDER BY created_at DESC'
  ).all(req.params.userId);
  res.json(vehicles);
});

// POST /api/vehicles - Thêm biển số mới
router.post('/', (req, res) => {
  const { user_id, plate_number } = req.body;
  if (!user_id || !plate_number)
    return res.status(400).json({ error: 'user_id và plate_number là bắt buộc' });

  const user = db.prepare('SELECT id FROM users WHERE id = ?').get(user_id);
  if (!user) return res.status(404).json({ error: 'User không tồn tại' });

  const result = db.prepare(
    'INSERT INTO vehicles (user_id, plate_number) VALUES (?, ?)'
  ).run(user_id, plate_number.toUpperCase());

  res.status(201).json(db.prepare('SELECT * FROM vehicles WHERE id = ?').get(result.lastInsertRowid));
});

// DELETE /api/vehicles/:id - Xóa biển số
router.delete('/:id', (req, res) => {
  const v = db.prepare('SELECT * FROM vehicles WHERE id = ?').get(req.params.id);
  if (!v) return res.status(404).json({ error: 'Không tìm thấy biển số' });

  db.prepare('DELETE FROM vehicles WHERE id = ?').run(v.id);
  res.json({ message: `Đã xóa biển số: ${v.plate_number}` });
});

module.exports = router;
