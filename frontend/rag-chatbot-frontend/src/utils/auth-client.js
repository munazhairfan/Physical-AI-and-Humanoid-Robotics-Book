/**
 * Docusaurus Auth Client
 *
 * This file provides authentication functions compatible with Docusaurus
 * that connect to your backend authentication service.
 */

import { authAPI } from './auth-api';

// useSession hook that connects to real auth API
const useSession = () => {
  // Check if user is authenticated by checking for token
  const isAuthenticated = authAPI.isAuthenticated();
  const user = authAPI.getUser();
  const token = authAPI.getToken();

  return {
    data: token && user ? { user, expires: null } : null,
    status: isAuthenticated ? 'authenticated' : 'unauthenticated',
    isLoading: false
  };
};

// signIn function that connects to real auth API
const signIn = async (provider, options = {}) => {
  try {
    if (provider === 'credentials') {
      if (options.email && options.password) {
        // Handle login
        if (options.name) {
          // This is a signup request
          const result = await authAPI.register({
            email: options.email,
            password: options.password,
            name: options.name
          });
          // Redirect to home page after successful signup
          if (typeof window !== 'undefined') {
            window.location.href = '/';
          }
          return { ok: true, user: result.user, redirect: true };
        } else {
          // This is a login request
          const result = await authAPI.login({
            email: options.email,
            password: options.password
          });
          // Redirect to home page after successful login
          if (typeof window !== 'undefined') {
            window.location.href = '/';
          }
          return { ok: true, user: result.user, redirect: true };
        }
      }
    } else if (provider === 'google') {
      // Handle Google OAuth - redirect to backend
      authAPI.loginWithGoogle();
      return { ok: true, redirect: true };
    } else if (provider === 'github') {
      // Handle GitHub OAuth - redirect to backend
      authAPI.loginWithGitHub();
      return { ok: true, redirect: true };
    }

    return { error: 'Invalid provider or credentials' };
  } catch (error) {
    console.error('Auth error:', error);
    return { error: error.message || 'Authentication failed' };
  }
};

// signOut function that connects to real auth API
const signOut = async () => {
  try {
    await authAPI.logout();
    return { ok: true };
  } catch (error) {
    console.error('Logout error:', error);
    return { ok: false, error: error.message };
  }
};

// Export auth functions
export { signIn, signOut, useSession };