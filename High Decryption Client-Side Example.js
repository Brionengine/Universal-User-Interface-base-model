const CryptoJS = require('crypto-js');

// Encrypt data using AES-256
function encryptData(data) {
    return CryptoJS.AES.encrypt(data, CryptoJS.enc.Utf8.parse(ENCRYPTION_KEY), {
        keySize: 256 / 32,
        iv: CryptoJS.lib.WordArray.random(128 / 8),
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7
    }).toString();
}

// Decrypt data using AES-256
function decryptData(encryptedData) {
    const bytes = CryptoJS.AES.decrypt(encryptedData, CryptoJS.enc.Utf8.parse(ENCRYPTION_KEY), {
        keySize: 256 / 32,
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7
    });
    return bytes.toString(CryptoJS.enc.Utf8);
}
