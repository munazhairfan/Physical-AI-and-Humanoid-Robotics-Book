import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// Robot-themed Feature Components
const RobotBrainSVG = () => (
  <svg className={styles.featureSvg} viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="#0f172a" opacity="0"/>
    <rect x="50" y="50" width="100" height="100" rx="15" fill="#1e293b" stroke="#475569" stroke-width="2"/>
    <path d="M60 60 H80 M60 70 H90 M60 80 H85 M70 90 H95 M80 100 H100 M90 110 H110 M100 120 H120"
          stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round" opacity="0.7"/>
    <circle cx="70" cy="70" r="4" fill="#06b6d4"/>
    <circle cx="130" cy="70" r="4" fill="#06b6d4"/>
    <circle cx="70" cy="130" r="4" fill="#06b6d4"/>
    <circle cx="130" cy="130" r="4" fill="#06b6d4"/>
    <circle cx="100" cy="100" r="6" fill="#ec4899"/>
    <line x1="70" y1="70" x2="130" y2="70" stroke="#8b5cf6" strokeWidth="1" opacity="0.5"/>
    <line x1="70" y1="70" x2="100" y2="100" stroke="#8b5cf6" strokeWidth="1" opacity="0.5"/>
    <line x1="130" y1="70" x2="100" y2="100" stroke="#8b5cf6" strokeWidth="1" opacity="0.5"/>
    <line x1="70" y1="130" x2="100" y2="100" stroke="#8b5cf6" strokeWidth="1" opacity="0.5"/>
    <line x1="130" y1="130" x2="100" y2="100" stroke="#8b5cf6" strokeWidth="1" opacity="0.5"/>
    <circle cx="85" cy="80" r="6" fill="#06b6d4"/>
    <circle cx="115" cy="80" r="6" fill="#06b6d4"/>
    <circle cx="85" cy="80" r="2" fill="#ffffff"/>
    <circle cx="115" cy="80" r="2" fill="#ffffff"/>
    <circle cx="100" cy="65" r="15" fill="#8b5cf6" opacity="0.1"/>
  </svg>
);

const HumanoidRobotSVG = () => (
  <svg className={styles.featureSvg} viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="#0f172a" opacity="0"/>
    <rect x="80" y="80" width="40" height="60" rx="5" fill="#334155" stroke="#475569" stroke-width="2"/>
    <circle cx="100" cy="60" r="20" fill="#1e293b" stroke="#475569" stroke-width="2"/>
    <rect x="50" y="90" width="30" height="10" rx="5" fill="#334155" stroke="#475569" stroke-width="2"/>
    <rect x="120" y="90" width="30" height="10" rx="5" fill="#334155" stroke="#475569" stroke-width="2"/>
    <rect x="85" y="140" width="10" height="30" rx="3" fill="#1e293b" stroke="#475569" stroke-width="2"/>
    <rect x="105" y="140" width="10" height="30" rx="3" fill="#1e293b" stroke="#475569" stroke-width="2"/>
    <circle cx="100" cy="90" r="8" stroke="#8b5cf6" strokeWidth="1" fill="none" opacity="0.5"/>
    <circle cx="100" cy="105" r="6" stroke="#06b6d4" strokeWidth="1" fill="none" opacity="0.5"/>
    <circle cx="100" cy="120" r="4" stroke="#ec4899" strokeWidth="1" fill="none" opacity="0.5"/>
    <circle cx="92" cy="58" r="3" fill="#06b6d4"/>
    <circle cx="108" cy="58" r="3" fill="#06b6d4"/>
    <rect x="85" y="95" width="30" height="15" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" opacity="0.7"/>
    <circle cx="100" cy="102" r="2" fill="#8b5cf6" opacity="0.8"/>
  </svg>
);

const RobotLabSVG = () => (
  <svg className={styles.featureSvg} viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="#0f172a" opacity="0"/>
    <rect x="30" y="130" width="140" height="15" fill="#334155" stroke="#475569" stroke-width="2"/>
    <rect x="40" y="145" width="10" height="40" fill="#1e293b" stroke="#475569" stroke-width="2"/>
    <rect x="150" y="145" width="10" height="40" fill="#1e293b" stroke="#475569" stroke-width="2"/>
    <line x1="100" y1="130" x2="100" y2="80" stroke="#475569" strokeWidth="8" strokeLinecap="round"/>
    <line x1="100" y1="80" x2="60" y2="60" stroke="#475569" strokeWidth="6" strokeLinecap="round"/>
    <line x1="60" y1="60" x2="40" y2="80" stroke="#475569" strokeWidth="6" strokeLinecap="round"/>
    <rect x="40" y="75" width="8" height="10" rx="2" fill="#1e293b" stroke="#475569" strokeWidth="1"/>
    <rect x="48" y="75" width="8" height="10" rx="2" fill="#1e293b" stroke="#475569" strokeWidth="1"/>
    <rect x="120" y="110" width="30" height="20" rx="5" fill="#1e293b" stroke="#475569" stroke-width="2"/>
    <circle cx="135" cy="120" r="6" stroke="#06b6d4" strokeWidth="2" fill="none" opacity="0.8"/>
    <rect x="150" y="100" width="20" height="30" rx="5" fill="#334155" stroke="#475569" stroke-width="2"/>
    <rect x="155" y="105" width="10" height="10" rx="2" fill="#8b5cf6" opacity="0.7"/>
    <rect x="60" y="100" width="15" height="25" rx="7" fill="none" stroke="#8b5cf6" strokeWidth="2" opacity="0.5"/>
    <rect x="57" y="120" width="20" height="8" rx="3" fill="#8b5cf6" opacity="0.2"/>
    <rect x="70" y="135" width="40" height="20" rx="3" fill="#334155" stroke="#475569" strokeWidth="1"/>
    <circle cx="75" cy="140" r="2" fill="#06b6d4"/>
    <circle cx="80" cy="145" r="2" fill="#ec4899"/>
    <circle cx="85" cy="140" r="2" fill="#8b5cf6"/>
    <circle cx="90" cy="145" r="2" fill="#06b6d4"/>
    <circle cx="95" cy="140" r="2" fill="#ec4899"/>
    <circle cx="100" cy="80" r="10" fill="#8b5cf6" opacity="0.1"/>
    <circle cx="135" cy="120" r="8" fill="#06b6d4" opacity="0.1"/>
  </svg>
);

type FeatureItem = {
  title: string;
  Svg: React.ComponentType;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: '🤖 Advanced AI Learning',
    Svg: RobotBrainSVG,
    description: (
      <>
        Explore cutting-edge concepts in artificial intelligence for robotics,
        from neural networks to machine learning algorithms that power autonomous machines.
      </>
    ),
  },
  {
    title: '🦾 Humanoid Robotics',
    Svg: HumanoidRobotSVG,
    description: (
      <>
        Understand the mechanics and control systems behind humanoid robots,
        including gait planning, balance control, and human-robot interaction.
      </>
    ),
  },
  {
    title: '🔬 Interactive Labs',
    Svg: RobotLabSVG,
    description: (
      <>
        Engaging hands-on experiments and simulations to deepen your understanding
        of robotics principles and physical AI concepts.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
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
