require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const CryptoJS = require('crypto-js');
const helmet = require('helmet');
const { check, validationResult } = require('express-validator');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use(helmet());

// Encryption key from environment variables
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY || 'DefaultSecretKey!';

// Route to handle command execution
app.post('/execute', [
    // Input validation
    check('command')
        .trim()
        .notEmpty()
        .withMessage('Command cannot be empty')
        .isAlphanumeric('en-US', { ignore: ' ' })
        .withMessage('Command contains invalid characters')
], (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
    }

    const { command } = req.body;

    // Encrypt the command
    const encryptedCommand = CryptoJS.AES.encrypt(command, ENCRYPTION_KEY).toString();

    // Log encrypted command securely
    console.log(`Encrypted Command: ${encryptedCommand}`);

    // Simulate command execution
    setTimeout(() => {
        const responseMessage = `Command "${command}" executed successfully.`;
        const encryptedResponse = CryptoJS.AES.encrypt(responseMessage, ENCRYPTION_KEY).toString();

        // Return encrypted response
        res.json({ message: encryptedResponse });
    }, 1000);
});

// Start the server
app.listen(PORT, () => {
    console.log(`BF UUI Server is running on http://localhost:${PORT}`);
});
