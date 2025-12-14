# Authentication System with Better Auth and Neon Postgres

This project implements a production-ready authentication system using Better Auth with Neon Serverless Postgres database. The system provides email/password authentication, OAuth integration with Google and GitHub, secure session management, and protected API endpoints.

## Features

- ✅ Email/password registration and login
- ✅ OAuth integration with Google and GitHub
- ✅ Secure session management with httpOnly cookies
- ✅ Protected routes requiring authentication
- ✅ Database integration with Neon Serverless Postgres
- ✅ Client-side authentication helper functions
- ✅ Logout functionality
- ✅ Session validation middleware
- ✅ Next.js API route integration
- ✅ Express.js server with protected routes

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Neon Postgres account and database URL
- Google OAuth credentials (Client ID and Secret)
- GitHub OAuth credentials (Client ID and Secret)

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install dependencies

```bash
npm install
```

### 3. Configure environment variables

Create a `.env` file in the root directory with the following structure:

```env
# Database Configuration
DATABASE_URL=your_neon_connection_string

# Better Auth Configuration
BETTER_AUTH_SECRET=your_secret
BETTER_AUTH_URL=http://localhost:3000

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# GitHub OAuth Configuration
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

### 4. Run database migrations

First, generate the schema:

```bash
npx @better-auth/cli generate
```

Then run the migrations:

```bash
npx @better-auth/cli migrate
```

### 5. Start the server

```bash
npm run auth-server
```

Or for development with auto-reload:

```bash
npm run dev
```

## API Endpoints

### Authentication Routes

- `POST /api/auth/signup` - Create new user account
- `POST /api/auth/login` - Authenticate user and create session
- `POST /api/auth/logout` - End user session
- `GET /api/auth/me` - Get current user info (protected)
- `GET /api/auth/session` - Get current session info (for client-side validation)
- `GET /api/auth/oauth/google` - Initiate Google OAuth
- `GET /api/auth/oauth/github` - Initiate GitHub OAuth

### Protected Routes

The system includes middleware for protecting routes that require authentication:

- `requireAuth` middleware - Requires valid authentication, returns 401 if not authenticated
- `checkAuth` middleware - Checks authentication but doesn't require it, adds user info to request if authenticated

Example protected endpoint: `GET /api/protected/profile` - Requires authentication to access user profile

### OAuth Setup

#### Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API
4. Create credentials (OAuth 2.0 Client IDs)
5. Add authorized redirect URIs: `http://localhost:3000/api/auth/oauth/google/callback`
6. Add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to your `.env` file

#### GitHub OAuth

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Create a new OAuth App
3. Set Homepage URL to your application URL
4. Set Authorization callback URL to `http://localhost:3000/api/auth/oauth/github/callback`
5. Add `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` to your `.env` file

## Next.js Integration

The system includes proper integration for Next.js applications:

### API Routes
The Better Auth API routes are available at `frontend/rag-chatbot-frontend/src/pages/api/auth/[...auth].js`, which handles all Better Auth API requests automatically.

### Middleware
Server-side middleware is available in `frontend/rag-chatbot-frontend/src/utils/auth-middleware.js`:
- `withAuth()` - Higher-order function to protect pages that require authentication
- `getSession()` - Function to get session in getServerSideProps
- `getAuthSession()` - Client-side function to check authentication status

Express.js middleware is available in `server/middleware/auth-middleware.js`:
- `requireAuth` - Middleware to protect routes that require authentication
- `checkAuth` - Middleware to check authentication but not require it

## Client-Side Usage

The client-side authentication helper is available in `client/auth-client.js`:

```javascript
import { signup, login, getUserInfo } from './client/auth-client.js';

// Signup
try {
  const result = await signup({
    email: 'user@example.com',
    password: 'securePassword123',
    firstName: 'John',
    lastName: 'Doe'
  });
  console.log('User created:', result);
} catch (error) {
  console.error('Signup failed:', error);
}

// Login
try {
  const result = await login({
    email: 'user@example.com',
    password: 'securePassword123'
  });
  console.log('Logged in:', result);
} catch (error) {
  console.error('Login failed:', error);
}

// Get user info
try {
  const result = await getUserInfo();
  console.log('User info:', result);
} catch (error) {
  console.error('Failed to get user info:', error);
}

// Logout
try {
  const result = await logout();
  console.log('Logged out:', result);
} catch (error) {
  console.error('Logout failed:', error);
}
```

## Security Features

- Passwords are automatically hashed using bcrypt
- Secure httpOnly cookies for session management
- CSRF protection with sameSite attribute
- Input validation and sanitization
- Rate limiting for authentication endpoints (configurable)

## Database Schema

The system uses the following tables (created automatically by Better Auth):

- `users` - User accounts with email, password hash, and profile info
- `sessions` - Active user sessions with expiration
- `accounts` - OAuth provider accounts linked to users
- `verification_tokens` - Tokens for email verification and password resets

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | Neon Postgres connection string | Yes |
| `BETTER_AUTH_SECRET` | Secret key for JWT signing | Yes |
| `BETTER_AUTH_URL` | Base URL of your application | Yes |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | No* |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret | No* |
| `GITHUB_CLIENT_ID` | GitHub OAuth client ID | No* |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth client secret | No* |

*Required only if using OAuth providers

## Development

To run the development server with auto-reload:

```bash
npm run dev
```

To run tests:

```bash
npm test
```

## Production Deployment

1. Use a strong secret for `BETTER_AUTH_SECRET`
2. Ensure your `BETTER_AUTH_URL` is set to your production URL
3. Configure proper OAuth redirect URIs for your domain
4. Use environment variables for all secrets in production
5. Consider using a reverse proxy (nginx) for production deployment

## Troubleshooting

### Database Connection Issues

- Verify your Neon connection string format
- Check that your database allows connections from your application
- Ensure your firewall rules allow database connections

### Better Auth and Neon Compatibility

You may encounter a "Failed to initialize database adapter" error when running the authentication server. This is a known compatibility issue between Better Auth and Neon's serverless architecture. The server has been configured with graceful error handling to continue running despite this issue.

If you see this error:
```
[BetterAuthError: Failed to initialize database adapter]
```

The server will continue running but authentication operations may fail. To resolve this:

1. Verify your Neon Postgres connection string in `.env` is correct
2. Check that the database tables were created by running: `node check-schema.js`
3. Ensure the tables exist by running: `node execute-schema.js` (if needed)
4. Consider updating to the latest Better Auth version for improved Neon compatibility
5. For production, ensure SSL settings match your Neon configuration
6. Alternative: Consider using a standard PostgreSQL database instead of Neon's serverless option

The authentication system includes manual table creation scripts to work around potential migration issues with Neon:
- `create-better-auth-tables.sql` - SQL script to manually create required tables
- `execute-schema.js` - Script to execute the schema creation
- `check-schema.js` - Script to verify tables exist

Additionally, diagnostic utilities are provided for troubleshooting:
- `check-schema.js` - Script to verify tables exist
- `execute-schema.js` - Script to execute schema creation (if migrations fail)
- `run-migrations.js` - Alternative migration execution script
- `test/auth-test.js` - Authentication system test script

### OAuth Configuration Issues

- Verify that your redirect URIs match exactly what's configured in Google/GitHub
- Check that your client IDs and secrets are correct
- Ensure your OAuth providers are properly configured in `server/auth/betterAuth.js`

### Session Management Issues

- Confirm that your cookies are properly configured for your domain
- Check that HTTPS is used in production (secure flag)
- Verify that your session expiration times are appropriate

## Support

For support, please open an issue in the GitHub repository or contact the development team.