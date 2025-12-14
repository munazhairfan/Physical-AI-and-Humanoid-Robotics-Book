import { auth } from '../../../../../../server/auth/betterAuth';

// Better Auth Next.js API route handler
export default async function handler(req, res) {
  // Use Better Auth's built-in handler for Next.js
  return auth.handler(req, res);
}

export const config = {
  api: {
    externalResolver: true, // Required for proper header handling
  },
};