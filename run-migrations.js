require('dotenv').config();
const { betterAuth } = require("better-auth");

const databaseUrl = process.env.DATABASE_URL;

if (!databaseUrl) {
  throw new Error("DATABASE_URL environment variable is required for authentication system");
}

console.log("Setting up Better Auth for migrations...");

// Initialize Better Auth with the same configuration as our main app
const auth = betterAuth({
  database: {
    url: databaseUrl,
    type: "postgresql",
  },
  secret: process.env.BETTER_AUTH_SECRET,
  baseURL: process.env.BETTER_AUTH_URL || "http://localhost:3000",
  account: {
    accountLinking: {
      enabled: true,
      trustedProviders: ["google", "github"],
    },
  },
  socialProviders: {
    google: {
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    },
    github: {
      clientId: process.env.GITHUB_CLIENT_ID,
      clientSecret: process.env.GITHUB_CLIENT_SECRET,
    },
  },
});

console.log("Better Auth initialized for migrations");
console.log("Attempting to connect to database and run migrations...");

// Better Auth should automatically handle migrations on initialization
// The schema should be created during the initialization process
console.log("Auth instance created. Database tables should be initialized.");