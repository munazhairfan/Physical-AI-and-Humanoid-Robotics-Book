import React from 'react';
import { useAuth, AuthGuard } from '../utils/auth';

const Profile = () => {
  const { user, isAuthenticated, logout } = useAuth();

  return (
    <AuthGuard fallback={<div>Please log in to view your profile.</div>}>
      <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
        <h1>Your Profile</h1>
        <div style={{
          backgroundColor: '#f8f9fa',
          padding: '1.5rem',
          borderRadius: '8px',
          border: '1px solid #e9ecef'
        }}>
          <h2>Welcome, {user?.name || user?.email}!</h2>
          <div style={{ marginBottom: '1rem' }}>
            <strong>Email:</strong> {user?.email}
          </div>
          <div style={{ marginBottom: '1rem' }}>
            <strong>User ID:</strong> {user?.id}
          </div>
          <button
            onClick={logout}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Logout
          </button>
        </div>
        <div style={{ marginTop: '2rem' }}>
          <h3>Account Features</h3>
          <ul>
            <li>Access to exclusive content</li>
            <li>Personalized recommendations</li>
            <li>Save your progress and bookmarks</li>
            <li>Personalized chat history</li>
          </ul>
        </div>
      </div>
    </AuthGuard>
  );
};

export default Profile;