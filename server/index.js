require('dotenv').config();
const express = require('express');
const cookieParser = require('cookie-parser');
const cors = require('cors');
// Handle unhandled promise rejections to prevent server crashes
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.error('This is likely due to Better Auth database connection issues');
  console.error('Server will continue running, but authentication operations may fail');
});

// Initialize auth routes with error handling to prevent server crash on startup
let authRoutes, protectedRoutes;
try {
  authRoutes = require('./routes/auth');
  protectedRoutes = require('./routes/protected');
  console.log('Authentication routes loaded successfully');
} catch (error) {
  console.error('Error loading authentication routes:', error.message);
  console.error('This may be due to Better Auth initialization issues');
  // Create placeholder routes that return error messages
  authRoutes = (req, res, next) => {
    res.status(500).json({
      error: 'Authentication system not available',
      message: 'Better Auth failed to initialize - check database connection'
    });
  };
  protectedRoutes = (req, res, next) => {
    res.status(500).json({
      error: 'Protected routes not available',
      message: 'Authentication system not initialized'
    });
  };
}

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:3000',
  credentials: true,
}));
app.use(express.json());
app.use(cookieParser());

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/protected', protectedRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK', message: 'Authentication server is running' });
});

// Root endpoint
app.get('/', (req, res) => {
  res.status(200).json({ message: 'Better Auth Server with Neon Postgres' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

app.listen(PORT, () => {
  console.log(`Authentication server is running on port ${PORT}`);
  console.log(`Health check available at http://localhost:${PORT}/health`);
  console.log(`Auth endpoints available at http://localhost:${PORT}/api/auth`);
  console.log(`Protected endpoints available at http://localhost:${PORT}/api/protected`);
  console.log(`
Note: If you see "Failed to initialize database adapter" errors,
the database tables have been created but Better Auth may need additional setup.`);
  console.log(`
Troubleshooting steps:
1. Verify your Neon Postgres connection string in .env is correct
2. Check that the database tables were created (run: node check-schema.js)
3. Consider updating to the latest Better Auth version
4. For production, ensure SSL settings match your Neon configuration
`);
});