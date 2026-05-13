const express = require('express');
const router = express.Router();
const db = require('../database');

// POST /api/auth/verify
// Body: { "username": "admin", "password": "123456" }
router.post('/verify', (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ granted: false, message: 'Missing username or password' });
  }

  const user = db.prepare(
    'SELECT * FROM users WHERE username = ? AND password = ?'
  ).get(username, password);

  if (user) {
    const vehicles = db.prepare(
      'SELECT plate_number FROM vehicles WHERE user_id = ?'
    ).all(user.id);

    return res.json({
      granted: true,
      message: 'Access granted',
      user: {
        id: user.id,
        username: user.username,
        full_name: user.full_name,
      },
      vehicles: vehicles.map(v => v.plate_number),
    });
  }

  return res.json({
    granted: false,
    message: 'Invalid username or password',
  });
});

module.exports = router;
