import type {ReactNode} from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import ChatWidget from '@site/src/components/ChatWidget';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics Assistant`}
      description="RAG Chatbot for robotics concepts and textbook questions">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <h2 style={{textAlign: 'center', marginTop: '2rem'}}>Physical AI & Humanoid Robotics Assistant</h2>
                <p style={{textAlign: 'center', marginBottom: '2rem'}}>
                  Ask questions about robotics concepts from the textbook. The AI will retrieve relevant information and provide detailed answers.
                </p>
              </div>
            </div>
            <div className="row">
              <div className="col col--12">
                <ChatWidget />
              </div>
            </div>
          </div>
        </section>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
