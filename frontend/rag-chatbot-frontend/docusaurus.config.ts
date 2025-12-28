import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Educational Book on Robotics and AI Concepts',
  favicon: 'img/master-light-switch-svgrepo-com.svg',


  // Set the production url of your site here (will be assigned by Vercel after first deployment)
  url: 'https://your-vercel-project-name.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For Vercel deployment, using root path ('/') is typically best for first deployment
  baseUrl: '/',

  // GitHub pages deployment config - not used for Vercel deployment
  organizationName: 'munazhairfan', // GitHub username/organization
  projectName: 'Physical-AI-and-Humanoid-Robotics-Book', // Repository name
  // Deployment settings not used for Vercel
  deploymentBranch: 'main',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
    localeConfigs: {
      en: {
        label: 'English',
      },
      ur: {
        label: 'اردو',
        htmlLang: 'ur'
      }
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/master-light-switch-svgrepo-com.svg', // Use the logo as social card
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI & Robotics',
      logo: {
        alt: 'Robotics Educational Book Logo',
        src: 'img/master-light-switch-svgrepo-com.svg',
      },
      items: [
        {
          to: '/',
          position: 'left',
          label: 'Home',
        },
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'Contents',
        },
        {
          to: '/login',
          label: 'Login',
          position: 'right',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/munazhairfan/Physical-AI-and-Humanoid-Robotics-Book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Educational Project`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,

  // Enhanced plugins for educational book functionality
  plugins: [
    './src/plugins/floatingChatPlugin', // Plugin to add the global floating chat component
  ],


  // Environment variables for deployment
  themes: [],
  stylesheets: [],
  scripts: [
    {
      src: '/js/selection-chatbot.js',
      async: true,
      defer: true,
    },
    {
      src: '/js/locale-docs-guard.js',
      defer: true,
    },
    {
      src: '/js/auth-pages-locale-guard.js',
      defer: true,
    },
  ],

  // Environment variables to be passed to the client bundle
  clientModules: [
    require.resolve('./src/clientModules/env.js'),
  ],
};

export default config;