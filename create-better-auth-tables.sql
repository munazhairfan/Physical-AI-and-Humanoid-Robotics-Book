-- Create Better Auth tables manually
-- These are the typical tables that Better Auth expects

-- Users table
CREATE TABLE IF NOT EXISTS "auth_user" (
  "id" TEXT PRIMARY KEY,
  "email" TEXT NOT NULL UNIQUE,
  "email_verified" BOOLEAN DEFAULT FALSE,
  "username" TEXT,
  "first_name" TEXT,
  "last_name" TEXT,
  "password" TEXT,
  "created_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE IF NOT EXISTS "auth_session" (
  "id" TEXT PRIMARY KEY,
  "user_id" TEXT NOT NULL REFERENCES "auth_user"("id") ON DELETE CASCADE,
  "expires_at" TIMESTAMP WITH TIME ZONE NOT NULL,
  "created_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Accounts table (for OAuth providers)
CREATE TABLE IF NOT EXISTS "auth_account" (
  "id" TEXT PRIMARY KEY,
  "user_id" TEXT NOT NULL REFERENCES "auth_user"("id") ON DELETE CASCADE,
  "provider_id" TEXT NOT NULL,
  "provider_account_id" TEXT NOT NULL,
  "access_token" TEXT,
  "refresh_token" TEXT,
  "expires_at" TIMESTAMP WITH TIME ZONE,
  "token_type" TEXT,
  "scope" TEXT,
  "created_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE("provider_id", "provider_account_id")
);

-- Verification tokens table
CREATE TABLE IF NOT EXISTS "auth_verification_token" (
  "id" TEXT PRIMARY KEY,
  "identifier" TEXT NOT NULL,
  "value" TEXT NOT NULL,
  "expires_at" TIMESTAMP WITH TIME ZONE NOT NULL,
  "type" TEXT NOT NULL,
  "created_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS "auth_user_email_idx" ON "auth_user"("email");
CREATE INDEX IF NOT EXISTS "auth_session_user_id_idx" ON "auth_session"("user_id");
CREATE INDEX IF NOT EXISTS "auth_account_user_id_idx" ON "auth_account"("user_id");
CREATE INDEX IF NOT EXISTS "auth_verification_token_identifier_idx" ON "auth_verification_token"("identifier");
CREATE INDEX IF NOT EXISTS "auth_verification_token_value_idx" ON "auth_verification_token"("value");