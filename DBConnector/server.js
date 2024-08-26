require('dotenv').config();
require('express-async-errors'); // Automatically handles async errors
const express = require('express');
const bodyParser = require('body-parser');
const { query } = require('./dbConnector');
const winston = require('winston');

const app = express();
const PORT = process.env.PORT || 3000;

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

app.use(bodyParser.json());

// Routes
app.get('/data', async (req, res) => {
    try {
        const results = await query('SELECT * FROM data_table');
        res.json(results);
    } catch (err) {
        logger.error(`Failed to fetch data: ${err.message}`);
        res.status(500).send('Failed to fetch data');
    }
});

app.post('/data', async (req, res) => {
    const { name, value } = req.body;
    if (!name || !value) {
        logger.warn('Invalid input detected');
        return res.status(400).send('Invalid input');
    }

    try {
        const results = await query('INSERT INTO data_table (name, value) VALUES (?, ?)', [name, value]);
        res.json({ success: true, id: results.insertId });
    } catch (err) {
        logger.error(`Failed to insert data: ${err.message}`);
        res.status(500).send('Failed to insert data');
    }
});

// Global error handler
app.use((err, req, res, next) => {
    logger.error(`Unhandled error: ${err.message}`);
    res.status(500).send('An unexpected error occurred');
});

app.listen(PORT, () => {
    logger.info(`BF UUI Server is running on http://localhost:${PORT}`);
});
