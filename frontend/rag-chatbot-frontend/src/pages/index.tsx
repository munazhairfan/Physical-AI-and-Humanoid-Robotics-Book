import type {ReactNode} from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import AnimatedBlobs from '@site/src/components/AnimatedBlobs/AnimatedBlobs';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <AnimatedBlobs />
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
              <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 18V5l12-2v13"/>
                <circle cx="6" cy="18" r="3"/>
                <circle cx="18" cy="16" r="3"/>
                <path d="M9 15l12-2"/>
              </svg>
            </div>
            <h3>Advanced AI Learning</h3>
            <p>Explore cutting-edge concepts in artificial intelligence for robotics, from neural networks to machine learning algorithms that power autonomous machines.</p>
          </div>
        </div>
        <div className="col col--4">
          <div className={styles.infoCard}>
            <div className={clsx(styles.cardIcon, styles.robotIcon)}>
              <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 8V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-2"/>
                <path d="M13 8h6"/>
                <path d="M16 13a4 4 0 1 1-8 0"/>
                <path d="M8 17v.01"/>
                <path d="M8 7v.01"/>
              </svg>
            </div>
            <h3>Humanoid Robotics</h3>
            <p>Understand the mechanics and control systems behind humanoid robots, including gait planning, balance control, and human-robot interaction.</p>
          </div>
        </div>
        <div className="col col--4">
          <div className={styles.infoCard}>
            <div className={clsx(styles.cardIcon, styles.robotIcon)}>
              <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 19V5"/>
                <path d="M5 12H2"/>
                <path d="M7 6H4"/>
                <path d="M7 18H4"/>
                <path d="M17 6v12"/>
                <path d="M22 12h-3"/>
                <path d="M19 6h-3"/>
                <path d="M19 18h-3"/>
              </svg>
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
