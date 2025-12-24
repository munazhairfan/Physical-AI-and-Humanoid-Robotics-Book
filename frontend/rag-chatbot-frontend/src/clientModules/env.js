// src/clientModules/env.js
// This module makes environment variables available to the client
// It runs before other modules are loaded

if (typeof window !== 'undefined') {
  // Define process.env if it doesn't exist
  if (typeof window.process === 'undefined') {
    window.process = {};
  }
  if (typeof window.process.env === 'undefined') {
    window.process.env = {};
  }

  // Set default environment variables for the client
  window.process.env.REACT_APP_API_URL = window.process.env.REACT_APP_API_URL || 'https://physical-ai-and-humanoid-robotics-book-production.up.railway.app';
}