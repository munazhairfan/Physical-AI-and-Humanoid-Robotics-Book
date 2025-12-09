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
          <div className="col col--10 col--offset--1"> {/* Increased width and adjusted offset for better centering */}
            <div className={clsx("text--center", styles.headerContent)}>
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

        {/* Adding a decorative robot illustration */}
        <div className={styles.robotIllustrationContainer}>
          <div className={styles.robotIllustration}>
            <img src="/img/Cool robot-bro.svg" alt="Robotic AI Illustration" className={styles.robotImage} />
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
              🦾
            </div>
            <h3>Advanced AI Learning</h3>
            <p>Explore cutting-edge concepts in artificial intelligence for robotics, from neural networks to machine learning algorithms that power autonomous machines.</p>
          </div>
        </div>
        <div className="col col--4">
          <div className={styles.infoCard}>
            <div className={clsx(styles.cardIcon, styles.robotIcon)}>
              🤖
            </div>
            <h3>Humanoid Robotics</h3>
            <p>Understand the mechanics and control systems behind humanoid robots, including gait planning, balance control, and human-robot interaction.</p>
          </div>
        </div>
        <div className="col col--4">
          <div className={styles.infoCard}>
            <div className={clsx(styles.cardIcon, styles.robotIcon)}>
              🔬
            </div>
            <h3>Interactive Labs</h3>
            <p>Engaging hands-on experiments and simulations to deepen your understanding of robotics principles and physical AI concepts.</p>
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
