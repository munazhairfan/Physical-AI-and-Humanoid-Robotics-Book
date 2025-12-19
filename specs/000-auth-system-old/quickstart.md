# Quickstart: Authentication System with Better Auth and Neon Postgres

## Overview
This guide provides step-by-step instructions to set up a production-ready authentication system using Better Auth with Neon Serverless Postgres database. The system includes email/password authentication, OAuth with Google and GitHub, and secure session management.

## Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Neon Postgres account and database URL
- Google OAuth credentials (Client ID and Secret)
- GitHub OAuth credentials (Client ID and Secret)

## Installation Steps

### 1. Install Dependencies
```bash
npm install better-auth @neondatabase/serverless pg express cookie-parser
npm install -D @better-auth/cli
```

### 2. Create Environment File
Create a `.env` file with the following structure:
```env
DATABASE_URL="your_neon_postgres_connection_string"
BETTER_AUTH_SECRET="your_secure_auth_secret"
BETTER_AUTH_URL="http://localhost:3000"

# Google OAuth
GOOGLE_CLIENT_ID="your_google_client_id"
GOOGLE_CLIENT_SECRET="your_google_client_secret"

# GitHub OAuth
GITHUB_CLIENT_ID="your_github_client_id"
GITHUB_CLIENT_SECRET="your_github_client_secret"
```

### 3. Initialize Better Auth
Create `server/auth/betterAuth.ts` with database connection and OAuth providers configuration.

### 4. Run Database Migrations
```bash
npx @better-auth/cli generate
npx @better-auth/cli migrate
```

### 5. Set Up Authentication Routes
Create API routes for signup, login, and protected endpoints using Better Auth's built-in functions.

## Configuration Details

### Database Setup
- Uses Neon Serverless Postgres for automatic scaling
- Connection pooling handled by the driver
- Secure connection with SSL enabled by default

### OAuth Providers
- Google OAuth configured with proper scopes
- GitHub OAuth configured with necessary permissions
- Callback URLs properly configured for both providers

### Session Management
- Secure httpOnly cookies for session storage
- Automatic session expiration
- Secure token generation and validation

## API Endpoints

### Authentication Routes
- `POST /api/auth/signup` - Create new user account
- `POST /api/auth/login` - Authenticate user and create session
- `GET /api/auth/me` - Get current user info (protected)

### OAuth Routes
- `GET /api/auth/oauth/google` - Initiate Google OAuth
- `GET /api/auth/oauth/github` - Initiate GitHub OAuth
- `GET /api/auth/oauth/callback` - Handle OAuth callbacks

## Security Features

### Password Security
- Automatic password hashing with bcrypt or similar
- Secure password requirements enforcement
- Protection against common password attacks

### Session Security
- httpOnly cookies prevent XSS attacks
- Secure flag ensures HTTPS transmission
- SameSite attribute prevents CSRF attacks
- Automatic session invalidation on logout

### Rate Limiting
- Built-in protection against brute force attacks
- Configurable rate limits for authentication endpoints

## Testing the Setup

### 1. Verify Database Connection
```bash
npx @better-auth/cli db:status
```

### 2. Test Authentication Flow
- Create a test user via signup endpoint
- Verify login works correctly
- Access protected endpoint with valid session

### 3. OAuth Flow Testing
- Test Google OAuth flow
- Test GitHub OAuth flow
- Verify account linking works properly

## Environment Variables

### Required Variables
- `DATABASE_URL`: Neon Postgres connection string
- `BETTER_AUTH_SECRET`: Secret key for JWT signing
- `BETTER_AUTH_URL`: Base URL of your application

### OAuth Provider Variables
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
- `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`

## Common Issues and Troubleshooting

### Database Connection Issues
- Verify Neon connection string format
- Check database permissions
- Ensure firewall rules allow connections

### OAuth Configuration Issues
- Verify redirect URIs match OAuth provider settings
- Check client IDs and secrets are correct
- Ensure proper OAuth scopes are requested

### Session Management Issues
- Confirm httpOnly and secure flags are properly set
- Verify cookie domain and path settings
- Check session expiration settings

## Next Steps

1. Implement password reset functionality
2. Add multi-factor authentication
3. Set up user profile management
4. Implement account verification emails
5. Add admin functionality and user management