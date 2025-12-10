// Floating Chat Plugin for Docusaurus
// This plugin adds a floating chat widget to all pages

const path = require('path');

module.exports = function (context) {
  const { siteConfig } = context;

  return {
    name: 'floating-chat-plugin',

    getClientModules() {
      const modulePath = path.resolve(__dirname, '../components/FloatingChatLoader');
      return [modulePath];
    },
  };
};