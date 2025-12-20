module.exports = function createWebpackConfigPlugin() {
  return {
    name: 'webpack-config-plugin',
    configureWebpack() {
      return {
        resolve: {
          fallback: {
            "crypto": require.resolve("crypto-browserify"),
            "http": require.resolve("stream-http"),
            "https": require.resolve("https-browserify"),
            "querystring": require.resolve("querystring-es3"),
            "zlib": require.resolve("browserify-zlib"),
            "stream": require.resolve("stream-browserify"),
            "buffer": require.resolve("buffer"),
            "util": require.resolve("util"),
            "url": require.resolve("url"),
            "assert": require.resolve("assert"),
            "path": require.resolve("path-browserify"),
            "process": require.resolve("process/browser"),
            "vm": require.resolve("vm-browserify"),
          }
        }
      };
    },
  };
};