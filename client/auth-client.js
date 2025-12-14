/**
 * Client-side authentication helper functions
 * Provides signup, login, and user info retrieval functions
 */

class AuthClient {
  constructor(baseURL = '/api/auth') {
    this.baseURL = baseURL;
  }

  /**
   * Signup with email and password
   * @param {Object} userData - User data including email and password
   * @returns {Promise<Object>} Response with user info and session
   */
  async signup(userData) {
    try {
      const response = await fetch(`${this.baseURL}/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `Signup failed: ${response.status}`);
      }

      return result;
    } catch (error) {
      console.error('Signup error:', error);
      throw error;
    }
  }

  /**
   * Login with email and password
   * @param {Object} credentials - Email and password
   * @returns {Promise<Object>} Response with user info and session
   */
  async login(credentials) {
    try {
      const response = await fetch(`${this.baseURL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `Login failed: ${response.status}`);
      }

      return result;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }

  /**
   * Get current user info
   * @returns {Promise<Object>} Response with user information
   */
  async getUserInfo() {
    try {
      const response = await fetch(`${this.baseURL}/me`, {
        method: 'GET',
        credentials: 'include', // Include cookies in the request
      });

      const result = await response.json();

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Unauthorized: Session expired or invalid');
        }
        throw new Error(result.error || `Failed to get user info: ${response.status}`);
      }

      return result;
    } catch (error) {
      console.error('Get user info error:', error);
      throw error;
    }
  }

  /**
   * Logout user
   * @returns {Promise<Object>} Response with logout confirmation
   */
  async logout() {
    try {
      const response = await fetch(`${this.baseURL}/logout`, {
        method: 'POST',
        credentials: 'include', // Include cookies in the request
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `Logout failed: ${response.status}`);
      }

      return result;
    } catch (error) {
      console.error('Logout error:', error);
      throw error;
    }
  }

  /**
   * Initiate Google OAuth flow
   */
  async initiateGoogleOAuth() {
    try {
      // Redirect to Google OAuth endpoint
      window.location.href = `${this.baseURL}/oauth/google`;
    } catch (error) {
      console.error('Google OAuth initiation error:', error);
      throw error;
    }
  }

  /**
   * Initiate GitHub OAuth flow
   */
  async initiateGitHubOAuth() {
    try {
      // Redirect to GitHub OAuth endpoint
      window.location.href = `${this.baseURL}/oauth/github`;
    } catch (error) {
      console.error('GitHub OAuth initiation error:', error);
      throw error;
    }
  }
}

// Export a singleton instance
const authClient = new AuthClient();
export default authClient;

// Also export individual functions for convenience
export const { signup, login, getUserInfo, logout, initiateGoogleOAuth, initiateGitHubOAuth } = authClient;