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
  // Enhanced sidebar for Physical AI & Humanoid Robotics Educational Book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      collapsed: false,
      items: [
        'intro',
      ],
    },
    {
      type: 'category',
      label: 'Foundations of Physical AI',
      collapsed: false,
      items: [
        'physical-ai-intro',
        'chapter1-introduction-physical-ai',
      ],
    },
    {
      type: 'category',
      label: 'Humanoid Robotics',
      collapsed: false,
      items: [
        'humanoid-robotics',
        'chapter2-humanoid-robotics-fundamentals',
      ],
    },
    {
      type: 'category',
      label: 'Sensing and Perception',
      collapsed: false,
      items: [
        'sensor-integration',
        'chapter3-sensor-integration-perception',
      ],
    },
    {
      type: 'category',
      label: 'AI and Control Systems',
      collapsed: false,
      items: [
        'chapter4-ai-in-humanoid-robotics',
      ],
    },
    {
      type: 'category',
      label: 'Applications and Use Cases',
      collapsed: false,
      items: [
        'chapter5-applications-humanoid-robots',
      ],
    },
    {
      type: 'category',
      label: 'Future Directions & Ethics',
      collapsed: false,
      items: [
        'chapter6-challenges-future-directions',
      ],
    },
  ],
};

export default sidebars;
