const express = require('express');
const { requireAuth } = require('../middleware/auth-middleware');

const router = express.Router();

// Example protected route that requires authentication
router.get('/profile', requireAuth, async (req, res) => {
  try {
    // req.user and req.session are available from the middleware
    res.status(200).json({
      user: req.user,
      message: "This is a protected route. You are authenticated.",
    });
  } catch (error) {
    console.error("Protected route error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Another example: protected API endpoint
router.get('/dashboard', requireAuth, async (req, res) => {
  try {
    // Access user info from the authenticated session
    const { id, email } = req.user;

    res.status(200).json({
      userId: id,
      userEmail: email,
      message: "Welcome to your dashboard!",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Dashboard route error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

module.exports = router;