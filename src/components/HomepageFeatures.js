import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI & Humanoid Robotics',
    description: (
      <>
        A comprehensive textbook covering the latest developments in physical AI and humanoid robotics.
        From perception and control to reinforcement learning and sensor fusion.
      </>
    ),
  },
  {
    title: 'Modular Learning',
    description: (
      <>
        Organized into 4 comprehensive modules that can be studied independently or as a complete course.
        Perfect for both beginners and advanced robotics researchers.
      </>
    ),
  },
  {
    title: 'Interactive Learning',
    description: (
      <>
        Integrated RAG chatbot provides instant answers to your robotics questions.
        Interactive diagrams and examples help you understand complex concepts.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}