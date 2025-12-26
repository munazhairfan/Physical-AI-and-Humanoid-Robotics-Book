/**
 * Docusaurus Auth Client
 *
 * This file provides authentication functions compatible with Docusaurus
 * that connect to your backend authentication service.
 */

import { authAPI } from './auth-api';

import { useState, useEffect } from 'react';

// useSession hook that connects to real auth API
const useSession = () => {
  const [sessionData, setSessionData] = useState({
    data: null,
    status: 'unauthenticated',
    isLoading: true,
    error: null
  });

  useEffect(() => {
    // Check if user is authenticated by checking for token
    // Only run on client side
    if (typeof window !== 'undefined') {
      try {
        const isAuthenticated = authAPI.isAuthenticated();
        const user = authAPI.getUser();
        const token = authAPI.getToken();

        setSessionData({
          data: token && user ? { user, expires: null } : null,
          status: isAuthenticated ? 'authenticated' : 'unauthenticated',
          isLoading: false,
          error: null
        });
      } catch (error) {
        setSessionData({
          data: null,
          status: 'unauthenticated',
          isLoading: false,
          error: error.message
        });
      }
    } else {
      // On server side, set default values
      setSessionData({
        data: null,
        status: 'unauthenticated',
        isLoading: false,
        error: null
      });
    }
  }, []);

  return sessionData;
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
      // Show user a notification before redirecting
      if (typeof window !== 'undefined') {
        // You can add a notification here if using a notification library
        console.log('Redirecting to Google for authentication...');
      }
      authAPI.loginWithGoogle();
      return { ok: true, redirect: true };
    } else if (provider === 'github') {
      // Handle GitHub OAuth - redirect to backend
      // Show user a notification before redirecting
      if (typeof window !== 'undefined') {
        console.log('Redirecting to GitHub for authentication...');
      }
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

// Function to verify email
const verifyEmail = async (token) => {
  try {
    const response = await fetch(`${typeof window !== 'undefined' ? window.process?.env?.REACT_APP_API_URL || process?.env?.REACT_APP_API_URL || 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app' : 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app'}/verify-email`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ token }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Email verification failed');
    }

    const data = await response.json();
    return { ok: true, user: data.user };
  } catch (error) {
    console.error('Email verification error:', error);
    return { ok: false, error: error.message };
  }
};

// Export auth functions
export { signIn, signOut, useSession, verifyEmail };