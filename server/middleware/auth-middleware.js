const { auth } = require('../auth/betterAuth'); // Adjust path as needed

/**
 * Middleware to protect routes that require authentication
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
async function requireAuth(req, res, next) {
  try {
    // Get session using Better Auth
    const session = await auth.api.getSession({
      headers: req.headers,
    });

    if (!session) {
      return res.status(401).json({
        error: "Unauthorized",
        code: "UNAUTHORIZED",
        message: "Authentication required to access this resource"
      });
    }

    // Add user info to request object for use in downstream handlers
    req.user = session.user;
    req.session = session.session;

    // Continue to the next middleware/route handler
    next();
  } catch (error) {
    console.error("Auth middleware error:", error);
    res.status(500).json({
      error: "Internal server error",
      code: "INTERNAL_ERROR"
    });
  }
}

/**
 * Middleware to check if user is authenticated but not require it
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
async function checkAuth(req, res, next) {
  try {
    // Get session using Better Auth (will be null if not authenticated)
    const session = await auth.api.getSession({
      headers: req.headers,
    });

    // Add user info to request object if authenticated, otherwise undefined
    req.user = session?.user;
    req.session = session?.session;

    // Continue to the next middleware/route handler
    next();
  } catch (error) {
    console.error("Check auth middleware error:", error);
    // Don't block the request, just don't set user info
    req.user = null;
    req.session = null;
    next();
  }
}

module.exports = {
  requireAuth,
  checkAuth
};