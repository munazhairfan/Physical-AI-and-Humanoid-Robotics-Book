# Tasks: Authentication System with Better Auth and Neon Postgres

**Feature**: Authentication System with Better Auth and Neon Postgres
**Branch**: `001-auth-system`
**Date**: 2025-12-13
**Spec**: `./specs/001-auth-system/spec.md`
**Plan**: `./specs/001-auth-system/plan.md`

## Dependencies

- User Story 1 (P1) must be completed before User Story 2 (P2)
- User Story 3 (P2) and User Story 4 (P2) can be developed in parallel after foundational setup
- User Story 5 (P1) depends on foundational authentication being implemented
- User Story 6 (P1) depends on foundational authentication being implemented

## Parallel Execution Examples

### Per Story:
- **P1**: Server setup [T001-T005] can run in parallel with environment setup [T006-T007]
- **P2**: OAuth Google [T013-T015] can run in parallel with OAuth GitHub [T016-T018]

## Implementation Strategy

**MVP Scope**: Complete User Story 1 (Email/Password Registration) and User Story 2 (Email/Password Login) with minimal viable functionality including basic authentication and session management.

**Incremental Delivery**:
1. MVP: Email/password auth with signup/login endpoints
2. Enhancement: OAuth integration with Google and GitHub
3. Integration: Protected routes and session management

---

## Phase 1: Setup

### Goal
Initialize project structure and install dependencies for the authentication system.

- [x] T001 Create `.env` file at project root with placeholders for production secrets
- [x] T002 [P] Install dependencies: better-auth @better-auth/cli @neondatabase/serverless pg express cookie-parser dotenv
- [x] T003 Verify package.json has all required dependencies installed
- [x] T004 [P] Create server directory structure: server/auth/, server/routes/, server/middleware/, server/config/
- [x] T005 Create README.md with setup instructions for Neon, environment variables, migrations, and running the server

## Phase 2: Foundational

### Goal
Implement foundational components that are prerequisites for all user stories.

- [x] T006 Configure Better Auth with Neon Postgres in `server/auth/betterAuth.js`
- [ ] T007 Initialize and run database migrations using Better Auth CLI
- [x] T008 [P] Implement OAuth providers using environment variables in betterAuth.js
- [x] T009 [P] Remove all fallback authentication logic from auth implementation
- [x] T010 Create `client/auth-client.js` snippet with signup, login, logout, and fetch user info functions
- [x] T011 Add structured error handling for auth flows in all routes
- [x] T012 Ensure compatibility with Next.js App Router by creating proper API route handler

## Phase 3: [US1] Email/Password Registration

### Goal
Implement email/password registration functionality with user creation and session management.

**Independent Test Criteria**: When user submits valid email and password, an account is created and user is logged in with secure session.

- [x] T013 Create POST /signup endpoint in `server/routes/auth.js`
- [x] T014 Implement user creation logic with email validation in signup endpoint
- [x] T015 Create user session and set httpOnly cookie in signup endpoint
- [x] T016 Validate email format according to RFC standards in signup endpoint
- [x] T017 Prevent duplicate email registration in signup endpoint
- [x] T018 Return user information and session data from signup endpoint

## Phase 4: [US2] Email/Password Login

### Goal
Implement email/password login functionality with session creation.

**Independent Test Criteria**: When user submits valid email and password, they are authenticated and logged in with secure session.

- [x] T019 Create POST /login endpoint in `server/routes/auth.js`
- [x] T020 Implement user verification logic with password authentication
- [x] T021 Create user session and set httpOnly cookie in login endpoint
- [x] T022 Handle incorrect password scenarios with proper error response
- [x] T023 Handle non-existent email scenarios with proper error response
- [x] T024 Return user information and session data from login endpoint

## Phase 5: [US3] OAuth Registration/Login with Google

### Goal
Implement Google OAuth integration for account creation and login.

**Independent Test Criteria**: When user completes Google OAuth flow, an account is created (if new) or authenticated (if existing) with secure session.

- [x] T025 Configure Google OAuth provider in Better Auth initialization
- [x] T026 Create GET /oauth/google endpoint to initiate Google OAuth flow
- [x] T027 Handle Google OAuth callback and user creation/login
- [x] T028 Link Google account to existing user if email matches
- [x] T029 Set secure session cookie after successful Google OAuth

## Phase 6: [US4] OAuth Registration/Login with GitHub

### Goal
Implement GitHub OAuth integration for account creation and login.

**Independent Test Criteria**: When user completes GitHub OAuth flow, an account is created (if new) or authenticated (if existing) with secure session.

- [x] T030 Configure GitHub OAuth provider in Better Auth initialization
- [x] T031 Create GET /oauth/github endpoint to initiate GitHub OAuth flow
- [x] T032 Handle GitHub OAuth callback and user creation/login
- [x] T033 Link GitHub account to existing user if email matches
- [x] T034 Set secure session cookie after successful GitHub OAuth

## Phase 7: [US5] Protected Route Access

### Goal
Implement protected route functionality that requires valid authentication.

**Independent Test Criteria**: When authenticated user accesses protected route, content is accessible; when unauthenticated user accesses protected route, they receive 401 error.

- [x] T035 Create GET /me endpoint in `server/routes/auth.js` to return user info
- [x] T036 Implement authentication middleware to validate sessions
- [x] T037 Return 401 Unauthorized for unauthenticated requests to protected routes
- [x] T038 Return user information for authenticated requests to /me endpoint
- [x] T039 Add middleware to protect sensitive routes requiring valid authentication

## Phase 8: [US6] Session Management

### Goal
Implement secure session management that persists across page refreshes and browser sessions.

**Independent Test Criteria**: When user logs in, session persists across page refreshes and browser sessions with appropriate expiration.

- [x] T040 Configure httpOnly cookies for secure session storage
- [x] T041 Set appropriate cookie expiration times for sessions
- [x] T042 Implement session validation and refresh logic
- [x] T043 Handle session expiration with proper user notification
- [x] T044 Test session persistence across page refreshes
- [x] T045 Test session persistence across browser sessions (if using persistent cookies)

## Phase 9: Polish & Cross-Cutting Concerns

### Goal
Finalize implementation with error handling, validation, and cross-cutting concerns.

- [x] T046 Add comprehensive error handling for database connection failures
- [ ] T047 Add rate limiting to prevent multiple simultaneous login attempts
- [x] T048 Implement proper validation for all input parameters
- [ ] T049 Add logging for security events and authentication attempts
- [x] T050 Update README with OAuth provider setup instructions
- [x] T051 Add validation that server starts successfully
- [x] T052 Test that signup/login works end-to-end
- [x] T053 Verify sessions persist properly across different scenarios
- [x] T054 Add client-side snippet documentation to README
- [x] T055 Add logout API route and client usage documentation
- [x] T056 Implement session validation middleware for Next.js App Router compatibility
- [x] T057 Perform final validation of all implemented functionality