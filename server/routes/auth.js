const express = require('express');
const { auth } = require('../auth/betterAuth'); // Import the auth instance

const router = express.Router();

// POST /signup - Create user and create session
router.post("/signup", async (req, res) => {
  try {
    const { email, password, firstName, lastName } = req.body;

    // Validate input
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    // Create user using Better Auth
    const user = await auth.api.signUp({
      body: {
        email,
        password,
        firstName: firstName || "",
        lastName: lastName || "",
      },
      headers: req.headers,
    });

    // Set session cookie
    if (user.session) {
      res.cookie("better-auth-session-token", user.session.token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "strict",
        maxAge: 24 * 60 * 60 * 1000, // 24 hours
        path: "/",
      });
    }

    res.status(200).json({
      user: user.user,
      session: user.session,
      message: "User created and logged in successfully",
    });
  } catch (error) {
    console.error("Signup error:", error);
    if (error.message?.includes("email")) {
      res.status(409).json({ error: "Email already exists", code: "EMAIL_EXISTS" });
    } else if (error.message?.includes("password")) {
      res.status(400).json({ error: "Invalid password format", code: "INVALID_PASSWORD" });
    } else {
      res.status(400).json({ error: error.message || "Signup failed", code: "SIGNUP_FAILED" });
    }
  }
});

// POST /login - Verify user and create session
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validate input
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    // Authenticate user using Better Auth
    const user = await auth.api.signIn({
      body: {
        email,
        password,
      },
      headers: req.headers,
    });

    // Set session cookie
    if (user.session) {
      res.cookie("better-auth-session-token", user.session.token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "strict",
        maxAge: 24 * 60 * 60 * 1000, // 24 hours
        path: "/",
      });
    }

    res.status(200).json({
      user: user.user,
      session: user.session,
      message: "Login successful",
    });
  } catch (error) {
    console.error("Login error:", error);
    if (error.message?.includes("invalid")) {
      res.status(401).json({ error: "Invalid credentials", code: "INVALID_CREDENTIALS" });
    } else {
      res.status(400).json({ error: error.message || "Login failed", code: "LOGIN_FAILED" });
    }
  }
});

// GET /me - Protected route returning user info
router.get("/me", async (req, res) => {
  try {
    // Get user from session using Better Auth
    const session = await auth.api.getSession({
      headers: req.headers,
    });

    if (!session) {
      return res.status(401).json({ error: "Unauthorized", code: "UNAUTHORIZED" });
    }

    res.status(200).json({
      user: session.user,
    });
  } catch (error) {
    console.error("Get user error:", error);
    res.status(400).json({ error: error.message || "Failed to get user", code: "GET_USER_FAILED" });
  }
});

// GET /oauth/google - Initiate Google OAuth flow
router.get("/oauth/google", async (req, res) => {
  try {
    // Redirect to Google OAuth using Better Auth
    const redirectUrl = await auth.social.google.getAuthorizationUrl({
      headers: req.headers,
    });

    res.redirect(redirectUrl);
  } catch (error) {
    console.error("Google OAuth error:", error);
    res.status(400).json({ error: error.message || "Google OAuth failed", code: "GOOGLE_OAUTH_FAILED" });
  }
});

// GET /oauth/github - Initiate GitHub OAuth flow
router.get("/oauth/github", async (req, res) => {
  try {
    // Redirect to GitHub OAuth using Better Auth
    const redirectUrl = await auth.social.github.getAuthorizationUrl({
      headers: req.headers,
    });

    res.redirect(redirectUrl);
  } catch (error) {
    console.error("GitHub OAuth error:", error);
    res.status(400).json({ error: error.message || "GitHub OAuth failed", code: "GITHUB_OAUTH_FAILED" });
  }
});

// POST /logout - End user session
router.post("/logout", async (req, res) => {
  try {
    // Get current session
    const session = await auth.api.getSession({
      headers: req.headers,
    });

    if (!session) {
      return res.status(401).json({ error: "No active session", code: "NO_SESSION" });
    }

    // Sign out user using Better Auth
    await auth.api.signOut({
      headers: req.headers,
    });

    // Clear session cookie
    res.clearCookie("better-auth-session-token", {
      path: "/",
    });

    res.status(200).json({
      message: "Logged out successfully",
    });
  } catch (error) {
    console.error("Logout error:", error);
    res.status(400).json({ error: error.message || "Logout failed", code: "LOGOUT_FAILED" });
  }
});

// GET /session - Get current session (for client-side validation)
router.get("/session", async (req, res) => {
  try {
    // Get user from session using Better Auth
    const session = await auth.api.getSession({
      headers: req.headers,
    });

    if (!session) {
      return res.status(401).json({ error: "Unauthorized", code: "UNAUTHORIZED" });
    }

    res.status(200).json({
      user: session.user,
      session: session.session,
    });
  } catch (error) {
    console.error("Get session error:", error);
    res.status(400).json({ error: error.message || "Failed to get session", code: "GET_SESSION_FAILED" });
  }
});

module.exports = router;