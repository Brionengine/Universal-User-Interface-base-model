require('dotenv').config();
const mysql = require('mysql2/promise');
const winston = require('winston');

// Setup winston logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'app.log' }),
        new winston.transports.Console()
    ],
});

const pool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || 'password',
    database: process.env.DB_NAME || 'bf_uui_db',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

async function query(sql, params) {
    try {
        const [results,] = await pool.execute(sql, params);
        return results;
    } catch (err) {
        logger.error(`Database query failed: ${err.message}`);
        throw err;
    }
}

module.exports = { query };
