/**
 * Docusaurus Authentication System
 *
 * This authentication system provides a framework compatible with Docusaurus
 * that can be easily connected to external auth services like Auth0, Clerk, or custom backends.
 */

// Auth context for React components
import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { useSession, signIn as authSignIn, signOut as authSignOut } from './auth-client';

const AuthContext = createContext();

// Auth reducer to manage auth state
const authReducer = (state, action) => {
  switch (action.type) {
    case 'LOGIN_START':
      return { ...state, loading: true, error: null };
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        loading: false,
        isAuthenticated: true,
        user: action.payload.user
      };
    case 'LOGIN_ERROR':
      return { ...state, loading: false, error: action.payload };
    case 'LOGOUT':
      return {
        ...state,
        isAuthenticated: false,
        user: null
      };
    case 'SET_USER':
      return {
        ...state,
        user: action.payload
      };
    case 'UPDATE_USER_VERIFICATION':
      return {
        ...state,
        user: {
          ...state.user,
          is_verified: action.payload.is_verified
        }
      };
    default:
      return state;
  }
};

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, {
    isAuthenticated: false,
    user: null,
    loading: false,
    error: null
  });

  // Use auth client's useSession hook to manage session
  const { data: session, status, isLoading, error: sessionError } = useSession();

  // Update local state when session changes
  useEffect(() => {
    if (!isLoading) {
      if (sessionError) {
        console.error('Auth session error:', sessionError);
        dispatch({ type: 'LOGOUT' });
      } else if (session && session.user) {
        dispatch({
          type: 'LOGIN_SUCCESS',
          payload: { user: session.user }
        });
      } else {
        dispatch({ type: 'LOGOUT' });
      }
    }
  }, [session, isLoading, sessionError]);

  // Login function using auth client
  const login = async (credentials) => {
    dispatch({ type: 'LOGIN_START' });

    try {
      const result = await authSignIn('credentials', {
        email: credentials.email,
        password: credentials.password,
      });

      if (result?.error) {
        throw new Error(result.error);
      }

      return { success: true, user: result.user };
    } catch (error) {
      dispatch({
        type: 'LOGIN_ERROR',
        payload: error.message
      });
      return { success: false, error: error.message };
    }
  };

  // Signup function using auth client
  const signup = async (userData) => {
    dispatch({ type: 'LOGIN_START' });

    try {
      const result = await authSignIn('credentials', {
        email: userData.email,
        password: userData.password,
        name: userData.name || userData.username || userData.email.split('@')[0]
      });

      if (result?.error) {
        throw new Error(result.error);
      }

      return { success: true, user: result.user };
    } catch (error) {
      dispatch({
        type: 'LOGIN_ERROR',
        payload: error.message
      });
      return { success: false, error: error.message };
    }
  };

  // Logout function using auth client
  const logout = async () => {
    try {
      await authSignOut();
      dispatch({ type: 'LOGOUT' });
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  // Get current user info
  const getCurrentUser = () => {
    return state.user;
  };

  // Check if user is authenticated
  const isAuthenticated = () => {
    return state.isAuthenticated;
  };

  const value = {
    ...state,
    loading: isLoading || state.loading,
    login,
    signup,
    logout,
    getCurrentUser,
    isAuthenticated
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Auth-aware components
export const AuthGuard = ({ children, fallback = null }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '200px'
      }}>
        <div>Loading...</div>
      </div>
    );
  }

  return isAuthenticated ? children : fallback;
};

export const LoginPage = () => {
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [name, setName] = React.useState('');
  const [isLogin, setIsLogin] = React.useState(true);
  const { login, signup, error } = useAuth();
  const [formError, setFormError] = React.useState('');
  const [successMessage, setSuccessMessage] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);

  // Validation functions
  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validatePassword = (password) => {
    // Password should be at least 8 characters with at least one uppercase, one lowercase, and one number
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(password);
  };

  const validateName = (name) => {
    // Name should be at least 2 characters and contain only letters, spaces, hyphens, and apostrophes
    const nameRegex = /^[a-zA-Z\s\-']+$/;
    return name.length >= 2 && nameRegex.test(name);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError('');
    setSuccessMessage('');
    setIsLoading(true);

    // Validate inputs before submitting
    if (!validateEmail(email)) {
      setFormError('Please enter a valid email address.');
      setIsLoading(false);
      return;
    }

    if (!validatePassword(password)) {
      setFormError('Password must be at least 8 characters with at least one uppercase letter, one lowercase letter, and one number.');
      setIsLoading(false);
      return;
    }

    if (!isLogin && !validateName(name)) {
      setFormError('Name must be at least 2 characters and contain only letters, spaces, hyphens, and apostrophes.');
      setIsLoading(false);
      return;
    }

    try {
      if (isLogin) {
        const result = await login({ email, password });
        if (!result.success) {
          // Improve error message for login failures
          let errorMessage = result.error;
          if (errorMessage.includes('401') || errorMessage.includes('incorrect')) {
            errorMessage = 'Incorrect email or password. Please try again.';
          } else if (errorMessage.includes('failed to fetch')) {
            errorMessage = 'Unable to connect to the server. Please check your internet connection and try again.';
          }
          setFormError(errorMessage);
        } else {
          setSuccessMessage('Login successful! Redirecting...');
          // Clear form after successful login
          setEmail('');
          setPassword('');
        }
      } else {
        const result = await signup({ email, password, name });
        if (!result.success) {
          // Improve error message for registration failures
          let errorMessage = result.error;
          if (errorMessage.includes('already registered')) {
            errorMessage = 'This email address is already registered. Please try logging in instead.';
          } else if (errorMessage.includes('failed to fetch')) {
            errorMessage = 'Unable to connect to the server. Please check your internet connection and try again.';
          }
          setFormError(errorMessage);
        } else {
          setSuccessMessage('Account created successfully! Welcome to our platform.');
          // Clear form after successful signup
          setEmail('');
          setPassword('');
          setName('');
        }
      }
    } catch (err) {
      setFormError('An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{
      maxWidth: '400px',
      margin: '2rem auto',
      padding: '2rem',
      border: '1px solid #ddd',
      borderRadius: '8px'
    }}>
      <h2>{isLogin ? 'Login' : 'Sign Up'}</h2>

      {/* Verification status indicator */}
      {isLogin && state.user && state.user.is_verified !== undefined && !state.user.is_verified && (
        <div style={{
          color: '#856404',
          marginBottom: '1rem',
          padding: '0.5rem',
          backgroundColor: '#fff3cd',
          border: '1px solid #ffeaa7',
          borderRadius: '4px'
        }}>
          ⚠️ Your email is not verified. Please check your email for verification instructions.
        </div>
      )}

      {/* Success message */}
      {successMessage && (
        <div style={{
          color: 'green',
          marginBottom: '1rem',
          padding: '0.5rem',
          backgroundColor: '#d4edda',
          border: '1px solid #c3e6cb',
          borderRadius: '4px'
        }}>
          {successMessage}
        </div>
      )}

      {/* Error message - show only one error at a time to avoid duplicates */}
      {(error || formError) && (
        <div style={{
          color: 'red',
          marginBottom: '1rem',
          padding: '0.5rem',
          backgroundColor: '#f8d7da',
          border: '1px solid #f5c6cb',
          borderRadius: '4px'
        }}>
          {formError || error}
        </div>
      )}

      {/* Informational message for first-time users */}
      {!isLogin && (
        <div style={{
          color: '#0c5460',
          marginBottom: '1rem',
          padding: '0.5rem',
          backgroundColor: '#d1ecf1',
          border: '1px solid #bee5eb',
          borderRadius: '4px',
          fontSize: '0.85rem'
        }}>
          After signing up, please check your email to verify your account.
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {!isLogin && (
          <div style={{ marginBottom: '1rem' }}>
            <input
              type="text"
              placeholder="Full Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required={!isLogin}
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #ccc',
                borderRadius: '4px'
              }}
            />
          </div>
        )}
        <div style={{ marginBottom: '1rem' }}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            style={{
              width: '100%',
              padding: '0.5rem',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          />
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{
              width: '100%',
              padding: '0.5rem',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          style={{
            width: '100%',
            padding: '0.75rem',
            backgroundColor: isLoading ? '#cccccc' : '#007cba',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
        >
          {isLoading ? (isLogin ? 'Logging in...' : 'Creating Account...') : (isLogin ? 'Login' : 'Sign Up')}
        </button>
      </form>

      <div style={{ marginTop: '1rem', textAlign: 'center' }}>
        <button
          onClick={() => {
            setIsLogin(!isLogin);
            setFormError('');
            setName('');
          }}
          style={{
            background: 'none',
            border: 'none',
            color: '#007cba',
            cursor: 'pointer',
            textDecoration: 'underline'
          }}
        >
          {isLogin ? "Don't have an account? Sign up" : "Already have an account? Login"}
        </button>
      </div>

      {/* Social login buttons */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '0.5rem' }}>Or continue with</div>
        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
          <button
            type="button"
            onClick={() => {
              if (typeof window !== 'undefined') {
                console.log('Redirecting to Google for authentication...');
                authSignIn('google');
              }
            }}
            style={{
              padding: '0.5rem 1rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              cursor: 'pointer',
              background: 'white',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            <span>🔍</span>
            <span>Google</span>
          </button>
          <button
            type="button"
            onClick={() => {
              if (typeof window !== 'undefined') {
                console.log('Redirecting to GitHub for authentication...');
                authSignIn('github');
              }
            }}
            style={{
              padding: '0.5rem 1rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              cursor: 'pointer',
              background: 'white',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            <span>🐱</span>
            <span>GitHub</span>
          </button>
        </div>
      </div>
    </div>
  );
};

// Utility function to get auth headers for API requests
export const addAuthHeader = (headers = {}) => {
  // In a real implementation, this would add auth headers
  // For now, it returns headers as-is
  return headers;
};

// Export default auth utility
export default {
  AuthProvider,
  useAuth,
  AuthGuard,
  LoginPage,
  addAuthHeader
};