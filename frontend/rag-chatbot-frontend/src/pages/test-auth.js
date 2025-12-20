import React, { useState } from 'react';
import { useAuth } from '../utils/auth';
import { signIn } from '../utils/auth-client';

const TestAuthPage = () => {
  const { user, isAuthenticated, login, logout, loading } = useAuth();
  const [email, setEmail] = useState('test@example.com');
  const [password, setPassword] = useState('password123');

  const handleLogin = async () => {
    const credentials = { email, password };
    await login(credentials);
  };

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Authentication Test Page</h1>

      {loading ? (
        <p>Loading...</p>
      ) : isAuthenticated ? (
        <div>
          <p>Welcome, {user?.name || user?.email}!</p>
          <p>User ID: {user?.id}</p>
          <button onClick={logout} style={{margin: '0.5rem', padding: '0.5rem 1rem', backgroundColor: '#007cba', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
            Logout
          </button>
        </div>
      ) : (
        <div>
          <p>You are not authenticated</p>
          <div style={{margin: '1rem 0'}}>
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{padding: '0.5rem', marginRight: '0.5rem'}}
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{padding: '0.5rem', marginRight: '0.5rem'}}
            />
            <button onClick={handleLogin} style={{padding: '0.5rem 1rem', backgroundColor: '#007cba', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
              Login
            </button>
          </div>
          <div>
            <button onClick={() => signIn('google')} style={{margin: '0.5rem', padding: '0.5rem 1rem', backgroundColor: '#4285F4', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
              Login with Google
            </button>
            <button onClick={() => signIn('github')} style={{margin: '0.5rem', padding: '0.5rem 1rem', backgroundColor: '#333', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer'}}>
              Login with GitHub
            </button>
          </div>
        </div>
      )}

      <div style={{ marginTop: '2rem' }}>
        <h3>Current User Info:</h3>
        <pre>{JSON.stringify(user, null, 2)}</pre>
      </div>
    </div>
  );
};

export default TestAuthPage;