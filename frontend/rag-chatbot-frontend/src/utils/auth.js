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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError('');

    if (isLogin) {
      const result = await login({ email, password });
      if (!result.success) {
        setFormError(result.error);
      }
    } else {
      const result = await signup({ email, password, name });
      if (!result.success) {
        setFormError(result.error);
      }
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

      {error && (
        <div style={{ color: 'red', marginBottom: '1rem' }}>
          {error}
        </div>
      )}

      {formError && (
        <div style={{ color: 'red', marginBottom: '1rem' }}>
          {formError}
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
          style={{
            width: '100%',
            padding: '0.75rem',
            backgroundColor: '#007cba',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          {isLogin ? 'Login' : 'Sign Up'}
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
            onClick={() => authSignIn('google')}
            style={{
              padding: '0.5rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              cursor: 'pointer',
              background: 'white'
            }}
          >
            Google
          </button>
          <button
            onClick={() => authSignIn('github')}
            style={{
              padding: '0.5rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              cursor: 'pointer',
              background: 'white'
            }}
          >
            GitHub
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