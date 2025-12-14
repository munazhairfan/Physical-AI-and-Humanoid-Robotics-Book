import { auth } from '../../../../../server/auth/betterAuth';

/**
 * Middleware to protect pages that require authentication
 * @param {function} handler - The page handler function
 * @returns {function} - The protected handler
 */
export function withAuth(handler) {
  return async (context) => {
    // Get the request from context (Next.js 12+)
    const req = context.req || context.request;

    try {
      // Verify session using Better Auth
      const session = await auth.api.getSession({
        headers: req.headers,
      });

      if (!session) {
        // Redirect to login page
        return {
          redirect: {
            destination: '/login', // Adjust this path as needed
            permanent: false,
          },
        };
      }

      // Add user info to context
      context.user = session.user;

      // Call the original handler
      return await handler(context);
    } catch (error) {
      console.error('Auth middleware error:', error);

      // Redirect to login page on error
      return {
        redirect: {
          destination: '/login', // Adjust this path as needed
          permanent: false,
        },
      };
    }
  };
}

/**
 * Server-side function to get session for getServerSideProps
 * @param {Object} context - Next.js context
 * @returns {Object} - Session object with user info or null
 */
export async function getSession(context) {
  const req = context.req || context.request;

  try {
    const session = await auth.api.getSession({
      headers: req.headers,
    });

    return session || null;
  } catch (error) {
    console.error('Get session error:', error);
    return null;
  }
}

/**
 * Client-side function to check if user is authenticated
 * @returns {Promise<Object>} - Session object with user info or null
 */
export async function getAuthSession() {
  try {
    const response = await fetch('/api/auth/session', {
      method: 'GET',
      credentials: 'include',
    });

    if (response.ok) {
      const data = await response.json();
      return data;
    }

    return null;
  } catch (error) {
    console.error('Get auth session error:', error);
    return null;
  }
}