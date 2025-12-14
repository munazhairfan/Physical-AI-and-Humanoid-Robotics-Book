// nodePolyfillPlugin.js - Plugin to handle Node.js built-in module imports
const NodePolyfillPlugin = require('node-polyfill-webpack-plugin');

module.exports = function (context, options) {
  return {
    name: 'node-polyfill-plugin',

    configureWebpack(config, isServer, content) {
      // Only configure for client-side builds
      if (!isServer) {
        config.plugins = config.plugins || [];
        config.plugins.push(new NodePolyfillPlugin());

        // Resolve node built-in modules
        config.resolve = config.resolve || {};
        config.resolve.fallback = {
          ...config.resolve.fallback,
          "module": false,
          "path": false,
          "fs": false,
          "os": false,
          "crypto": false,
          "stream": false,
          "buffer": false,
          "process": false
        };
      }
      return config;
    },
  };
};