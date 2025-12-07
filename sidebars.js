// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro/preface',
        'intro/how-to-use'
      ],
    },
    {
      type: 'category',
      label: 'Module 1 — Perception & Computer Vision',
      items: [
        'module-1/overview',
        'module-1/core-concepts',
        'module-1/architecture',
        'module-1/algorithms',
        'module-1/assignments',
        'module-1/summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 2 — Robotic Nervous System (ROS2)',
      items: [
        'module-2/overview',
        'module-2/core-concepts',
        'module-2/architecture',
        'module-2/nodes-topics-services',
        'module-2/qos-dds',
        'module-2/assignments',
        'module-2/summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 3 — AI Perception & Sensor Fusion',
      items: [
        'module-3/overview',
        'module-3/core-concepts',
        'module-3/sensors',
        'module-3/perception-algorithms',
        'module-3/fusion-techniques',
        'module-3/assignments',
        'module-3/summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 4 — Reinforcement Learning & Control',
      items: [
        'module-4/overview',
        'module-4/rl-basics',
        'module-4/advanced-techniques',
        'module-4/control-integration',
        'module-4/assignments',
        'module-4/summary',
      ],
    },
    {
      type: 'category',
      label: 'RAG Chatbot',
      items: [
        'rag-chatbot/embedding',
        'rag-chatbot/api',
        'rag-chatbot/ui-integration',
        'rag-chatbot/deployment',
      ],
    },
    {
      type: 'category',
      label: 'Appendix',
      items: [
        'appendix/glossary',
        'appendix/references',
      ],
    },
  ],
};

module.exports = sidebars;