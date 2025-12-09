import type {ReactNode} from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset--2">
            <div className="text--center">
              <div className={styles.heroLogo}>
                <img src="/img/logo.svg" alt="Robotics Educational Book Logo" className={styles.heroImage} />
              </div>
              <div className={styles.titleContainer}>
                <Heading as="h1" className="hero__title">
                  {siteConfig.title}
                </Heading>
              </div>
              <p className="hero__subtitle">{siteConfig.tagline}</p>
              <div className={styles.heroButtons}>
                <button className={styles.primaryButton}>Start Learning</button>
                <button className={styles.secondaryButton}>Explore Topics</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function InfoCards() {
  return (
    <div className="container margin-vert--lg">
      <div className="row">
        <div className="col col--4">
          <div className={styles.infoCard}>
            <div className={clsx(styles.cardIcon, styles.robotIcon)}>
              🤖
            </div>
            <h3>Physical AI Fundamentals</h3>
            <p>Learn the core concepts of AI systems that interact with the physical world through perception, reasoning, and action.</p>
          </div>
        </div>
        <div className="col col--4">
          <div className={styles.infoCard}>
            <div className={clsx(styles.cardIcon, styles.robotIcon)}>
              🦾
            </div>
            <h3>Humanoid Robotics</h3>
            <p>Explore the design and control principles of robots that mimic human form for operation in human environments.</p>
          </div>
        </div>
        <div className="col col--4">
          <div className={styles.infoCard}>
            <div className={clsx(styles.cardIcon, styles.robotIcon)}>
              🧠
            </div>
            <h3>AI Control Systems</h3>
            <p>Discover how artificial intelligence algorithms enable adaptive behavior in robotic systems.</p>
          </div>
        </div>
      </div>
    </div>
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
        <InfoCards />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
