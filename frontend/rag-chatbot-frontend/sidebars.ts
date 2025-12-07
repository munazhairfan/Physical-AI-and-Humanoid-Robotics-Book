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
      items: [
        'physical-ai-intro',
        'chapter1-introduction-physical-ai'
      ],
    },
    {
      type: 'category',
      label: 'Humanoid Robotics Fundamentals',
      items: [
        'humanoid-robotics',
        'chapter2-humanoid-robotics-fundamentals'
      ],
    },
    {
      type: 'category',
      label: 'Sensors & Perception',
      items: [
        'sensor-integration',
        'chapter3-sensor-integration-perception'
      ],
    },
    {
      type: 'category',
      label: 'AI in Robotics',
      items: [
        'chapter4-ai-in-humanoid-robotics'
      ],
    },
    {
      type: 'category',
      label: 'Applications',
      items: [
        'chapter5-applications-humanoid-robots'
      ],
    },
    {
      type: 'category',
      label: 'Challenges & Future',
      items: [
        'chapter6-challenges-future-directions'
      ],
    },
  ],
};

export default sidebars;
