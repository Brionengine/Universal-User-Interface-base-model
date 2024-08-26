// Example function for key rotation
function rotateEncryptionKey() {
    // Generate a new key and update your .env file securely
    // In production, use a proper secrets management solution
    const newKey = CryptoJS.lib.WordArray.random(256 / 8).toString();
    console.log(`New Encryption Key: ${newKey}`);
    // Store newKey securely and update all encrypted data as needed
}
