import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs',
    component: ComponentCreator('/docs', '905'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '8e0'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '575'),
            routes: [
              {
                path: '/docs/appendix/glossary',
                component: ComponentCreator('/docs/appendix/glossary', '96f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/appendix/references',
                component: ComponentCreator('/docs/appendix/references', '589'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/intro/how-to-use',
                component: ComponentCreator('/docs/intro/how-to-use', '605'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/intro/preface',
                component: ComponentCreator('/docs/intro/preface', '4ab'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-1/algorithms',
                component: ComponentCreator('/docs/module-1/algorithms', '315'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-1/architecture',
                component: ComponentCreator('/docs/module-1/architecture', '851'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-1/assignments',
                component: ComponentCreator('/docs/module-1/assignments', 'd00'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-1/core-concepts',
                component: ComponentCreator('/docs/module-1/core-concepts', 'd54'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-1/diagrams/action-workflow',
                component: ComponentCreator('/docs/module-1/diagrams/action-workflow', '623'),
                exact: true
              },
              {
                path: '/docs/module-1/diagrams/multi-robot-comm',
                component: ComponentCreator('/docs/module-1/diagrams/multi-robot-comm', 'c74'),
                exact: true
              },
              {
                path: '/docs/module-1/diagrams/service-handshake',
                component: ComponentCreator('/docs/module-1/diagrams/service-handshake', '53f'),
                exact: true
              },
              {
                path: '/docs/module-1/diagrams/topic-flow',
                component: ComponentCreator('/docs/module-1/diagrams/topic-flow', '0cf'),
                exact: true
              },
              {
                path: '/docs/module-1/examples/launch-files',
                component: ComponentCreator('/docs/module-1/examples/launch-files', '1c6'),
                exact: true
              },
              {
                path: '/docs/module-1/examples/publisher-subscriber',
                component: ComponentCreator('/docs/module-1/examples/publisher-subscriber', 'd2a'),
                exact: true
              },
              {
                path: '/docs/module-1/examples/services-actions',
                component: ComponentCreator('/docs/module-1/examples/services-actions', '6b2'),
                exact: true
              },
              {
                path: '/docs/module-1/overview',
                component: ComponentCreator('/docs/module-1/overview', '9ef'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-1/summary',
                component: ComponentCreator('/docs/module-1/summary', 'fd4'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/architecture',
                component: ComponentCreator('/docs/module-2/architecture', '4ea'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/assignments',
                component: ComponentCreator('/docs/module-2/assignments', 'f8e'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/core-concepts',
                component: ComponentCreator('/docs/module-2/core-concepts', '843'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/deep-vision',
                component: ComponentCreator('/docs/module-2/deep-vision', 'ec9'),
                exact: true
              },
              {
                path: '/docs/module-2/examples/camera-stream',
                component: ComponentCreator('/docs/module-2/examples/camera-stream', '3b7'),
                exact: true
              },
              {
                path: '/docs/module-2/examples/image-filtering',
                component: ComponentCreator('/docs/module-2/examples/image-filtering', '4eb'),
                exact: true
              },
              {
                path: '/docs/module-2/examples/object-detection',
                component: ComponentCreator('/docs/module-2/examples/object-detection', 'e76'),
                exact: true
              },
              {
                path: '/docs/module-2/fundamentals',
                component: ComponentCreator('/docs/module-2/fundamentals', '0bc'),
                exact: true
              },
              {
                path: '/docs/module-2/image-processing',
                component: ComponentCreator('/docs/module-2/image-processing', '164'),
                exact: true
              },
              {
                path: '/docs/module-2/mermaid-setup',
                component: ComponentCreator('/docs/module-2/mermaid-setup', 'a57'),
                exact: true
              },
              {
                path: '/docs/module-2/nodes-topics-services',
                component: ComponentCreator('/docs/module-2/nodes-topics-services', 'ea0'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/overview',
                component: ComponentCreator('/docs/module-2/overview', '9e1'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/qos-dds',
                component: ComponentCreator('/docs/module-2/qos-dds', 'e63'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/sensors',
                component: ComponentCreator('/docs/module-2/sensors', '9d3'),
                exact: true
              },
              {
                path: '/docs/module-2/setup-ros2',
                component: ComponentCreator('/docs/module-2/setup-ros2', '60f'),
                exact: true
              },
              {
                path: '/docs/module-2/summary',
                component: ComponentCreator('/docs/module-2/summary', 'b0c'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-2/validation-tools',
                component: ComponentCreator('/docs/module-2/validation-tools', '0fa'),
                exact: true
              },
              {
                path: '/docs/module-3/assignments',
                component: ComponentCreator('/docs/module-3/assignments', '9fc'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-3/core-concepts',
                component: ComponentCreator('/docs/module-3/core-concepts', '0bc'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-3/examples/camera-lidar-fusion',
                component: ComponentCreator('/docs/module-3/examples/camera-lidar-fusion', 'e8f'),
                exact: true
              },
              {
                path: '/docs/module-3/examples/object-detection',
                component: ComponentCreator('/docs/module-3/examples/object-detection', '075'),
                exact: true
              },
              {
                path: '/docs/module-3/examples/sensor-calibration',
                component: ComponentCreator('/docs/module-3/examples/sensor-calibration', 'd2f'),
                exact: true
              },
              {
                path: '/docs/module-3/fusion-techniques',
                component: ComponentCreator('/docs/module-3/fusion-techniques', 'bc7'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-3/overview',
                component: ComponentCreator('/docs/module-3/overview', 'f1d'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-3/perception-algorithms',
                component: ComponentCreator('/docs/module-3/perception-algorithms', '728'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-3/sensors',
                component: ComponentCreator('/docs/module-3/sensors', '40e'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-3/summary',
                component: ComponentCreator('/docs/module-3/summary', '825'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-4/advanced-techniques',
                component: ComponentCreator('/docs/module-4/advanced-techniques', 'c36'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-4/assignments',
                component: ComponentCreator('/docs/module-4/assignments', 'b0f'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-4/control-integration',
                component: ComponentCreator('/docs/module-4/control-integration', '519'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-4/control-systems',
                component: ComponentCreator('/docs/module-4/control-systems', '1d9'),
                exact: true
              },
              {
                path: '/docs/module-4/kinematics',
                component: ComponentCreator('/docs/module-4/kinematics', 'da8'),
                exact: true
              },
              {
                path: '/docs/module-4/motion-planning-algorithms',
                component: ComponentCreator('/docs/module-4/motion-planning-algorithms', 'e5a'),
                exact: true
              },
              {
                path: '/docs/module-4/overview',
                component: ComponentCreator('/docs/module-4/overview', '859'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-4/rl-basics',
                component: ComponentCreator('/docs/module-4/rl-basics', '003'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-4/summary',
                component: ComponentCreator('/docs/module-4/summary', 'edb'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/module-4/trajectory-generation',
                component: ComponentCreator('/docs/module-4/trajectory-generation', '6a2'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/api',
                component: ComponentCreator('/docs/rag-chatbot/api', '56c'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/rag-chatbot/architecture',
                component: ComponentCreator('/docs/rag-chatbot/architecture', '9e4'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/database-neon',
                component: ComponentCreator('/docs/rag-chatbot/database-neon', 'd31'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/deployment',
                component: ComponentCreator('/docs/rag-chatbot/deployment', 'e26'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/rag-chatbot/diagrams/backend-endpoints',
                component: ComponentCreator('/docs/rag-chatbot/diagrams/backend-endpoints', 'ad0'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/diagrams/data-flow',
                component: ComponentCreator('/docs/rag-chatbot/diagrams/data-flow', '83f'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/diagrams/frontend-component-interaction',
                component: ComponentCreator('/docs/rag-chatbot/diagrams/frontend-component-interaction', '46d'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/diagrams/overall-architecture',
                component: ComponentCreator('/docs/rag-chatbot/diagrams/overall-architecture', '12a'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/embedding',
                component: ComponentCreator('/docs/rag-chatbot/embedding', 'c1a'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/rag-chatbot/examples/backend-setup',
                component: ComponentCreator('/docs/rag-chatbot/examples/backend-setup', 'ffb'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/examples/chat-interface',
                component: ComponentCreator('/docs/rag-chatbot/examples/chat-interface', '6f0'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/examples/embed-document',
                component: ComponentCreator('/docs/rag-chatbot/examples/embed-document', 'b88'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/examples/query-vectorstore',
                component: ComponentCreator('/docs/rag-chatbot/examples/query-vectorstore', 'b0e'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/examples/selected-text-capture',
                component: ComponentCreator('/docs/rag-chatbot/examples/selected-text-capture', '144'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/mcp-integration',
                component: ComponentCreator('/docs/rag-chatbot/mcp-integration', 'd57'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/overview',
                component: ComponentCreator('/docs/rag-chatbot/overview', '7f0'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/summary',
                component: ComponentCreator('/docs/rag-chatbot/summary', '00f'),
                exact: true
              },
              {
                path: '/docs/rag-chatbot/ui-integration',
                component: ComponentCreator('/docs/rag-chatbot/ui-integration', '83b'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/docs/rag-chatbot/vectorstore-qdrant',
                component: ComponentCreator('/docs/rag-chatbot/vectorstore-qdrant', 'a0f'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
