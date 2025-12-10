import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import FloatingChat from '../components/ChatWidget/FloatingChat';

// Simple floating chat button that integrates with the full chat functionality
const SimpleFloatingChat = () => {
  return <FloatingChat />;
};

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.headerContent}>
          <div className={styles.heroLogo}>
            <img src="/img/chatbot.svg" alt="Robotics Educational Book Logo" className={styles.heroImage} />
          </div>
          <div className={styles.titleContainer}>
            <Heading as="h1" className={styles.hero__title}>
              {siteConfig.title}
            </Heading>
          <p className={styles.hero__subtitle}>{siteConfig.tagline}</p>
          </div>
          <div className={styles.heroButtons}>
            <Link to="/docs/intro" className={styles.primaryButton}>Start Learning</Link>
            <Link to="/docs/intro" className={styles.secondaryButton}>Explore Topics</Link>
          </div>
        </div>
      </div>
    </header>
  );
}


export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="Educational Book on Robotics Concepts and AI Integration">
      <HomepageHeader />
      <main>
        {/* Add any additional content here if needed */}
      </main>
    </Layout>
  );
}
