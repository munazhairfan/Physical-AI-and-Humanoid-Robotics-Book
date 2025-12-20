/**
 * Real Authentication API Service for Docusaurus
 *
 * This service connects to your backend authentication API
 */

const API_BASE_URL =
  typeof window !== 'undefined'
    ? (window.process?.env?.REACT_APP_API_URL || process?.env?.REACT_APP_API_URL || 'http://localhost:8000')
    : 'http://localhost:8000';

// Helper function to get auth token from localStorage (with SSR check)
const getAuthToken = () => {
  if (typeof window !== 'undefined' && window.localStorage) {
    return localStorage.getItem('auth_token');
  }
  return null;
};

// Helper function to set auth token in localStorage (with SSR check)
const setAuthToken = (token) => {
  if (typeof window !== 'undefined' && window.localStorage) {
    localStorage.setItem('auth_token', token);
  }
};

// Helper function to remove auth token (with SSR check)
const removeAuthToken = () => {
  if (typeof window !== 'undefined' && window.localStorage) {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
  }
};

// Helper function to get user from localStorage (with SSR check)
const getUser = () => {
  if (typeof window !== 'undefined' && window.localStorage) {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
  }
  return null;
};

// Helper function to set user in localStorage (with SSR check)
const setUser = (user) => {
  if (typeof window !== 'undefined' && window.localStorage) {
    localStorage.setItem('user', JSON.stringify(user));
  }
};

// Helper function to remove user (with SSR check)
const removeUser = () => {
  if (typeof window !== 'undefined' && window.localStorage) {
    localStorage.removeItem('user');
  }
};

// Helper function to add auth headers to requests
const getAuthHeaders = () => {
  const token = getAuthToken();
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` })
  };
};

// API functions
export const authAPI = {
  // Register a new user
  async register(userData) {
    try {
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Registration failed');
      }

      if (data.token) {
        setAuthToken(data.token);
        setUser(data.user);
      }

      return data;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  },

  // Login user
  async login(credentials) {
    try {
      const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      if (data.token) {
        setAuthToken(data.token);
        setUser(data.user);
      }

      return data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },

  // Login with Google (redirect to backend OAuth)
  async loginWithGoogle() {
    // For OAuth, redirect to backend endpoint
    window.location.href = `${API_BASE_URL}/auth/google`;
  },

  // Login with GitHub (redirect to backend OAuth)
  async loginWithGitHub() {
    // For OAuth, redirect to backend endpoint
    window.location.href = `${API_BASE_URL}/auth/github`;
  },

  // Get current user info
  async getCurrentUser() {
    try {
      const token = getAuthToken();
      if (!token) {
        return null;
      }

      const response = await fetch(`${API_BASE_URL}/me`, {
        method: 'GET',
        headers: getAuthHeaders(),
      });

      if (!response.ok) {
        // If unauthorized, remove stored tokens
        if (response.status === 401) {
          removeAuthToken();
          removeUser();
        }
        throw new Error('Failed to get user info');
      }

      const data = await response.json();
      setUser(data.user);
      return data.user;
    } catch (error) {
      console.error('Get user error:', error);
      return null;
    }
  },

  // Logout user
  async logout() {
    try {
      const token = getAuthToken();
      if (token) {
        await fetch(`${API_BASE_URL}/logout`, {
          method: 'POST',
          headers: getAuthHeaders(),
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
      // Continue with local cleanup even if API call fails
    } finally {
      removeAuthToken();
      removeUser();
    }
  },

  // Check if user is authenticated
  isAuthenticated() {
    return !!getAuthToken();
  },

  // Get token
  getToken() {
    return getAuthToken();
  },

  // Get user
  getUser() {
    return getUser();
  }
};