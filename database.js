const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

// Tạo thư mục data nếu chưa có
const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir);

const db = new Database(path.join(dataDir, 'parking.db'));

// Bật WAL mode cho performance tốt hơn
db.pragma('journal_mode = WAL');

// ======== TẠO BẢNG ========
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT    NOT NULL UNIQUE,
    password   TEXT    NOT NULL,
    full_name  TEXT    DEFAULT '',
    created_at DATETIME DEFAULT (datetime('now','localtime'))
  );

  CREATE TABLE IF NOT EXISTS vehicles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    plate_number TEXT    NOT NULL,
    created_at   DATETIME DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS access_logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER,
    username  TEXT    NOT NULL,
    plate_number TEXT DEFAULT '',
    granted   INTEGER NOT NULL DEFAULT 0,
    status    TEXT    DEFAULT 'IN', -- 'IN' hoặc 'OUT'
    timestamp DATETIME DEFAULT (datetime('now','localtime')),
    note      TEXT    DEFAULT ''
  );
`);

// Migration: Thêm cột status nếu bảng đã tồn tại từ trước mà chưa có cột này
try {
  db.prepare("ALTER TABLE access_logs ADD COLUMN status TEXT DEFAULT 'IN'").run();
  console.log('[DB] Added column status to access_logs');
} catch (err) {
  // Column đã tồn tại, không sao cả
}

// ======== SEED DATA MẪU ========
try {
  db.prepare("ALTER TABLE access_logs ADD COLUMN plate_number TEXT DEFAULT ''").run();
  console.log('[DB] Added column plate_number to access_logs');
} catch (err) {
  // Column da ton tai, khong sao ca
}

const existing = db.prepare('SELECT COUNT(*) as cnt FROM users').get();
if (existing.cnt === 0) {
  const insertUser = db.prepare(
    'INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)'
  );
  const u1 = insertUser.run('admin',  '123456', 'Quản trị viên');
  const u2 = insertUser.run('user1',  '123456', 'Nguyễn Văn A');
  const u3 = insertUser.run('user2',  'abcdef', 'Trần Thị B');

  const insertVehicle = db.prepare(
    'INSERT INTO vehicles (user_id, plate_number) VALUES (?, ?)'
  );
  insertVehicle.run(u1.lastInsertRowid, '51A-12345');
  insertVehicle.run(u1.lastInsertRowid, '99E122268');
  insertVehicle.run(u2.lastInsertRowid, '59B-67890');
  insertVehicle.run(u2.lastInsertRowid, '30C-11111');
  insertVehicle.run(u2.lastInsertRowid, '51F97022');
  insertVehicle.run(u3.lastInsertRowid, '43D-99999');

  console.log('[DB] Seed data created');
}

console.log('[DB] SQLite ready:', path.join(dataDir, 'parking.db'));

module.exports = db;
