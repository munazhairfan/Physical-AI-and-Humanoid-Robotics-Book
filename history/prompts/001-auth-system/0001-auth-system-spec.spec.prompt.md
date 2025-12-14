---
id: 0001
title: Auth System Spec
stage: spec
date: 2025-12-13
surface: agent
model: Claude Sonnet 4.5
feature: 001-auth-system
branch: 001-auth-system
user: user
command: /sp.specify
labels: [feature, specification]
links:
  spec: ./specs/001-auth-system/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - ./specs/001-auth-system/spec.md
tests:
  - npm run build (frontend)
  - python -m pytest (backend)
---

## Prompt

Project: Real authentication system for production using Better Auth and Neon Serverless Postgres.

GOALS:
1. Configure Better Auth for real production use.
2. Use Neon Serverless Postgres as the database.
3. Implement authentication features:
   - Email/password sign-up and login
   - OAuth providers: Google and GitHub
   - Session handling with secure httpOnly cookies
4. Run database migrations to create required tables in Neon Postgres.
5. Provide server-side routes for:
   - /api/auth/signup
   - /api/auth/login
   - /api/auth/me (protected route)
6. Provide client-side snippet to call auth endpoints.
7. Create `.env` placeholders only; instruct user to fill with actual secrets.
8. Provide commands to install dependencies, generate schema, run migrations, and start server.

CONSTRAINTS:
- Use real database connection (`DATABASE_URL`) to Neon.
- Initialize Better Auth with `BETTER_AUTH_SECRET`.
- Implement session-based authentication.
- Avoid any placeholder or "example" logic in routes.

## Response snapshot

Successfully created comprehensive specification for authentication system using Better Auth and Neon Serverless Postgres. The specification includes:

- 6 prioritized user stories covering core authentication functionality
- 16 functional requirements including OAuth integration, session management, and protected routes
- 3 key entities (User, Session, OAuth Account)
- 8 measurable success criteria

The specification covers all requested functionality including email/password auth, OAuth with Google and GitHub, secure session handling, database integration with Neon Postgres, and proper API endpoints.

## Outcome

- ✅ Impact: Complete specification ready for planning and implementation
- 🧪 Tests: Specification covers testable user scenarios and acceptance criteria
- 📁 Files: spec.md created with comprehensive requirements
- 🔁 Next prompts: Ready for /sp.plan to create implementation plan
- 🧠 Reflection: Specification addresses all core auth requirements with proper security considerations

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - All requirements captured
- Prompt variant (if applicable): Standard specification workflow
- Next experiment (smallest change to try): Begin implementation planning with /sp.plan
