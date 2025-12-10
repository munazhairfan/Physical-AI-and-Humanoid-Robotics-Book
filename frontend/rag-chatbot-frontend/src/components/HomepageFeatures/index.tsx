import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// Anime-themed Feature Components
const AnimeBrainSVG = () => (
  <svg className={styles.featureSvg} width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2C8.13401 2 5 5.13401 5 9C5 10.05 5.28 11.03 5.78 11.87L3 22H5.5L7.02 16H16.98L18.5 22H21L18.22 11.87C18.72 11.03 19 10.05 19 9C19 5.13401 15.866 2 12 2Z" fill="url(#brain_gradient)"/>
    <path d="M12 6C10.3431 6 9 7.34315 9 9C9 10.6569 10.3431 12 12 12C13.6569 12 15 10.6569 15 9C15 7.34315 13.6569 6 12 6Z" fill="url(#brain_inner_gradient)"/>
    <defs>
      <linearGradient id="brain_gradient" x1="5" y1="2" x2="19" y2="22" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#ffbe0b"/>
      </linearGradient>
      <linearGradient id="brain_inner_gradient" x1="9" y1="6" x2="15" y2="12" gradientUnits="userSpaceOnUse">
        <stop stopColor="#4ecdc4"/>
        <stop offset="1" stopColor="#ff6b6b"/>
      </linearGradient>
    </defs>
  </svg>
);

const AnimeRobotSVG = () => (
  <svg className={styles.featureSvg} width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M8 4V2H16V4H8Z" fill="url(#robot_headband)"/>
    <rect x="5" y="4" width="14" height="16" rx="2" fill="url(#robot_body)"/>
    <rect x="7" y="6" width="10" height="12" rx="1" fill="url(#robot_chest)"/>
    <circle cx="9" cy="8" r="1" fill="url(#robot_eye)"/>
    <circle cx="15" cy="8" r="1" fill="url(#robot_eye)"/>
    <rect x="10" y="10" width="4" height="1" rx="0.5" fill="url(#robot_mouth)"/>
    <path d="M4 8L2 10V14L4 16" stroke="url(#robot_arm)" strokeWidth="2"/>
    <path d="M20 8L22 10V14L20 16" stroke="url(#robot_arm)" strokeWidth="2"/>
    <path d="M8 20L6 22H18L16 20" stroke="url(#robot_leg)" strokeWidth="2"/>
    <defs>
      <linearGradient id="robot_headband" x1="8" y1="2" x2="16" y2="4" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#ffbe0b"/>
      </linearGradient>
      <linearGradient id="robot_body" x1="5" y1="4" x2="19" y2="20" gradientUnits="userSpaceOnUse">
        <stop stopColor="#4ecdc4"/>
        <stop offset="1" stopColor="#ff6b6b"/>
      </linearGradient>
      <linearGradient id="robot_chest" x1="7" y1="6" x2="17" y2="18" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ffbe0b"/>
        <stop offset="1" stopColor="#4ecdc4"/>
      </linearGradient>
      <linearGradient id="robot_eye" x1="8" y1="7" x2="10" y2="9" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#ffbe0b"/>
      </linearGradient>
      <linearGradient id="robot_mouth" x1="10" y1="10" x2="14" y2="11" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#ffbe0b"/>
      </linearGradient>
      <linearGradient id="robot_arm" x1="2" y1="10" x2="22" y2="10" gradientUnits="userSpaceOnUse">
        <stop stopColor="#4ecdc4"/>
        <stop offset="1" stopColor="#ff6b6b"/>
      </linearGradient>
      <linearGradient id="robot_leg" x1="6" y1="22" x2="18" y2="22" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#4ecdc4"/>
      </linearGradient>
    </defs>
  </svg>
);

const AnimeLabSVG = () => (
  <svg className={styles.featureSvg} width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="3" y="10" width="18" height="10" rx="2" fill="url(#lab_table)"/>
    <circle cx="8" cy="14" r="1.5" fill="url(#lab_beaker1)"/>
    <circle cx="12" cy="14" r="1.5" fill="url(#lab_beaker2)"/>
    <circle cx="16" cy="14" r="1.5" fill="url(#lab_beaker3)"/>
    <path d="M8 12.5L8 9" stroke="url(#lab_beaker1)" strokeWidth="1.5"/>
    <path d="M12 12.5L12 9" stroke="url(#lab_beaker2)" strokeWidth="1.5"/>
    <path d="M16 12.5L16 9" stroke="url(#lab_beaker3)" strokeWidth="1.5"/>
    <path d="M10 6L14 6" stroke="url(#lab_microscope)" strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 6L12 3" stroke="url(#lab_microscope)" strokeWidth="2" strokeLinecap="round"/>
    <circle cx="12" cy="2.5" r="1.5" fill="url(#lab_microscope_lens)"/>
    <rect x="1" y="20" width="22" height="2" fill="url(#lab_floor)"/>
    <defs>
      <linearGradient id="lab_table" x1="3" y1="10" x2="21" y2="20" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ffbe0b"/>
        <stop offset="1" stopColor="#4ecdc4"/>
      </linearGradient>
      <linearGradient id="lab_beaker1" x1="6.5" y1="12.5" x2="9.5" y2="15.5" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#ffbe0b"/>
      </linearGradient>
      <linearGradient id="lab_beaker2" x1="10.5" y1="12.5" x2="13.5" y2="15.5" gradientUnits="userSpaceOnUse">
        <stop stopColor="#4ecdc4"/>
        <stop offset="1" stopColor="#ff6b6b"/>
      </linearGradient>
      <linearGradient id="lab_beaker3" x1="14.5" y1="12.5" x2="17.5" y2="15.5" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ffbe0b"/>
        <stop offset="1" stopColor="#4ecdc4"/>
      </linearGradient>
      <linearGradient id="lab_microscope" x1="10" y1="3" x2="14" y2="6" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#ffbe0b"/>
      </linearGradient>
      <linearGradient id="lab_microscope_lens" x1="10.5" y1="1" x2="13.5" y2="4" gradientUnits="userSpaceOnUse">
        <stop stopColor="#4ecdc4"/>
        <stop offset="1" stopColor="#ff6b6b"/>
      </linearGradient>
      <linearGradient id="lab_floor" x1="1" y1="20" x2="23" y2="22" gradientUnits="userSpaceOnUse">
        <stop stopColor="#ff6b6b"/>
        <stop offset="1" stopColor="#4ecdc4"/>
      </linearGradient>
    </defs>
  </svg>
);

type FeatureItem = {
  title: string;
  Svg: React.ComponentType;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Advanced AI Learning',
    Svg: AnimeBrainSVG,
    description: (
      <>
        Explore cutting-edge concepts in artificial intelligence for robotics,
        from neural networks to machine learning algorithms that power autonomous machines.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    Svg: AnimeRobotSVG,
    description: (
      <>
        Understand the mechanics and control systems behind humanoid robots,
        including gait planning, balance control, and human-robot interaction.
      </>
    ),
  },
  {
    title: 'Interactive Labs',
    Svg: AnimeLabSVG,
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
