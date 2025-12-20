/**
 * Root-level wrapper for the Docusaurus application
 * Provides the AuthProvider context to the entire app
 */

import React from 'react';
import { AuthProvider } from '../utils/auth';

export default function Root({ children }) {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
}