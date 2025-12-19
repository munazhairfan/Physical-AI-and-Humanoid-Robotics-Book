# Data Model: Authentication System with Better Auth and Neon Postgres

## User Entity
- **id**: string (primary key, UUID)
- **email**: string (unique, required, validated format)
- **emailVerified**: boolean (default: false)
- **firstName**: string (optional)
- **lastName**: string (optional)
- **password**: string (hashed, required for email/password users)
- **createdAt**: timestamp (auto-generated)
- **updatedAt**: timestamp (auto-generated)
- **role**: string (default: "user", for future role-based access)
- **isActive**: boolean (default: true)

## Session Entity
- **id**: string (primary key, UUID)
- **userId**: string (foreign key to User)
- **token**: string (unique, encrypted session token)
- **expiresAt**: timestamp (session expiration time)
- **createdAt**: timestamp (auto-generated)
- **updatedAt**: timestamp (auto-generated)
- **userAgent**: string (optional, for device tracking)
- **ipAddress**: string (optional, for security tracking)

## OAuth Account Entity
- **id**: string (primary key, UUID)
- **userId**: string (foreign key to User, nullable for account linking)
- **provider**: string (enum: "google", "github", etc.)
- **providerAccountId**: string (unique per provider)
- **accessToken**: string (encrypted)
- **refreshToken**: string (encrypted, optional)
- **expiresAt**: timestamp (token expiration, optional)
- **createdAt**: timestamp (auto-generated)
- **updatedAt**: timestamp (auto-generated)
- **email**: string (from OAuth provider)
- **firstName**: string (from OAuth provider, optional)
- **lastName**: string (from OAuth provider, optional)

## Database Relationships
- User (1) → (M) Session (one-to-many: user has many sessions)
- User (1) → (M) OAuth Account (one-to-many: user has many OAuth accounts)
- OAuth Account (1) → (1) User (optional: unlinked OAuth accounts)

## Indexes
- User.email: unique index for fast login lookup
- Session.token: unique index for session validation
- OAuthAccount.provider + OAuthAccount.providerAccountId: composite unique index
- Session.expiresAt: index for automatic cleanup

## Validation Rules
- User.email: must be valid email format, unique
- User.password: minimum length requirements, proper hashing
- Session.expiresAt: must be in the future
- OAuthAccount.provider: must be one of supported providers
- All timestamps: automatically managed by database

## Security Constraints
- Passwords: never stored in plain text, always hashed
- Session tokens: encrypted and rotated appropriately
- OAuth tokens: encrypted at rest
- User data: proper sanitization before storage