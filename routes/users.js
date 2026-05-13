const express = require('express');
const router  = express.Router();
const db      = require('../database');

// GET /api/users - Danh sách tất cả users
router.get('/', (req, res) => {
  const users = db.prepare(`
    SELECT u.*, COUNT(v.id) as vehicle_count
    FROM users u
    LEFT JOIN vehicles v ON v.user_id = u.id
    GROUP BY u.id
    ORDER BY u.created_at DESC
  `).all();
  res.json(users);
});

// GET /api/users/:id - Chi tiết 1 user
router.get('/:id', (req, res) => {
  const user = db.prepare('SELECT * FROM users WHERE id = ?').get(req.params.id);
  if (!user) return res.status(404).json({ error: 'User not found' });

  const vehicles = db.prepare('SELECT * FROM vehicles WHERE user_id = ?').all(user.id);
  res.json({ ...user, vehicles });
});

// POST /api/users - Tạo user mới
router.post('/', (req, res) => {
  const { username, password, full_name } = req.body;
  if (!username || !password)
    return res.status(400).json({ error: 'username và password là bắt buộc' });

  try {
    const result = db.prepare(
      'INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)'
    ).run(username, password, full_name || '');

    const user = db.prepare('SELECT * FROM users WHERE id = ?').get(result.lastInsertRowid);
    res.status(201).json(user);
  } catch (e) {
    if (e.message.includes('UNIQUE')) {
      return res.status(409).json({ error: 'Username đã tồn tại' });
    }
    res.status(500).json({ error: e.message });
  }
});

// PUT /api/users/:id - Cập nhật user
router.put('/:id', (req, res) => {
  const { password, full_name } = req.body;
  const user = db.prepare('SELECT * FROM users WHERE id = ?').get(req.params.id);
  if (!user) return res.status(404).json({ error: 'User not found' });

  db.prepare(
    'UPDATE users SET password = ?, full_name = ? WHERE id = ?'
  ).run(
    password  || user.password,
    full_name !== undefined ? full_name : user.full_name,
    user.id
  );

  res.json(db.prepare('SELECT * FROM users WHERE id = ?').get(user.id));
});

// DELETE /api/users/:id - Xóa user (cascade xóa vehicles)
router.delete('/:id', (req, res) => {
  const user = db.prepare('SELECT * FROM users WHERE id = ?').get(req.params.id);
  if (!user) return res.status(404).json({ error: 'User not found' });

  db.prepare('DELETE FROM users WHERE id = ?').run(user.id);
  res.json({ message: `Đã xóa user: ${user.username}` });
});

module.exports = router;
