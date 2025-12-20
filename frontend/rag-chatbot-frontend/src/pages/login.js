import React from 'react';
import { LoginPage } from '../utils/auth';

const Login = () => {
  return (
    <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
      <h1>Login to Your Account</h1>
      <p>Access exclusive content and features by logging in to your account.</p>
      <LoginPage />
    </div>
  );
};

export default Login;