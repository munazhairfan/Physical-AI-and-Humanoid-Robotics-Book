import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for robotics content
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Physical AI Concepts',
      items: ['physical-ai-intro'],
    },
    {
      type: 'category',
      label: 'Humanoid Robotics',
      items: ['humanoid-robotics'],
    },
    {
      type: 'category',
      label: 'Sensors & Perception',
      items: ['sensor-integration'],
    },
  ],
};

export default sidebars;
