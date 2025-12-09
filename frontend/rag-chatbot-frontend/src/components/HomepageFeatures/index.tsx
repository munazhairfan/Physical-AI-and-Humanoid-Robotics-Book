import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// Robot-themed Feature Components
const RobotBrainSVG = () => (
  <svg className={styles.featureSvg} width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g clipPath="url(#clip0_1_2)">
      <path d="M12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2Z" stroke="url(#paint0_linear_1_2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M8 10H16" stroke="url(#paint1_linear_1_2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M8 14H13" stroke="url(#paint2_linear_1_2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M10 8V16" stroke="url(#paint3_linear_1_2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M14 8V16" stroke="url(#paint4_linear_1_2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </g>
    <defs>
      <linearGradient id="paint0_linear_1_2" x1="2" y1="12" x2="22" y2="12" gradientUnits="userSpaceOnUse">
        <stop stopColor="#EC4899"/>
        <stop offset="1" stopColor="#F59E0B"/>
      </linearGradient>
      <linearGradient id="paint1_linear_1_2" x1="8" y1="10" x2="16" y2="10" gradientUnits="userSpaceOnUse">
        <stop stopColor="#8B5CF6"/>
        <stop offset="1" stopColor="#EC4899"/>
      </linearGradient>
      <linearGradient id="paint2_linear_1_2" x1="8" y1="14" x2="13" y2="14" gradientUnits="userSpaceOnUse">
        <stop stopColor="#06B6D4"/>
        <stop offset="1" stopColor="#3B82F6"/>
      </linearGradient>
      <linearGradient id="paint3_linear_1_2" x1="10" y1="8" x2="10" y2="16" gradientUnits="userSpaceOnUse">
        <stop stopColor="#F59E0B"/>
        <stop offset="1" stopColor="#FBBF24"/>
      </linearGradient>
      <linearGradient id="paint4_linear_1_2" x1="14" y1="8" x2="14" y2="16" gradientUnits="userSpaceOnUse">
        <stop stopColor="#EC4899"/>
        <stop offset="1" stopColor="#F43F5E"/>
      </linearGradient>
      <clipPath id="clip0_1_2">
        <rect width="24" height="24" fill="white"/>
      </clipPath>
    </defs>
  </svg>
);

const HumanoidRobotSVG = () => (
  <svg className={styles.featureSvg} width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g clipPath="url(#clip0_1_4)">
      <path d="M4 17C4 15.3431 5.34315 14 7 14H17C18.6569 14 20 15.3431 20 17V20C20 21.1046 19.1046 22 18 22H6C4.89543 22 4 21.1046 4 20V17Z" stroke="url(#paint0_linear_1_4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 14V10" stroke="url(#paint1_linear_1_4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M8 10H16" stroke="url(#paint2_linear_1_4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 6C13.1046 6 14 5.10457 14 4C14 2.89543 13.1046 2 12 2C10.89543 2 10 2.89543 10 4C10 5.10457 10.89543 6 12 6Z" stroke="url(#paint3_linear_1_4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </g>
    <defs>
      <linearGradient id="paint0_linear_1_4" x1="4" y1="14" x2="20" y2="14" gradientUnits="userSpaceOnUse">
        <stop stopColor="#EC4899"/>
        <stop offset="1" stopColor="#F59E0B"/>
      </linearGradient>
      <linearGradient id="paint1_linear_1_4" x1="12" y1="10" x2="12" y2="14" gradientUnits="userSpaceOnUse">
        <stop stopColor="#06B6D4"/>
        <stop offset="1" stopColor="#3B82F6"/>
      </linearGradient>
      <linearGradient id="paint2_linear_1_4" x1="8" y1="10" x2="16" y2="10" gradientUnits="userSpaceOnUse">
        <stop stopColor="#8B5CF6"/>
        <stop offset="1" stopColor="#EC4899"/>
      </linearGradient>
      <linearGradient id="paint3_linear_1_4" x1="10" y1="2" x2="14" y2="2" gradientUnits="userSpaceOnUse">
        <stop stopColor="#F59E0B"/>
        <stop offset="1" stopColor="#FBBF24"/>
      </linearGradient>
      <clipPath id="clip0_1_4">
        <rect width="24" height="24" fill="white"/>
      </clipPath>
    </defs>
  </svg>
);

const RobotLabSVG = () => (
  <svg className={styles.featureSvg} width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g clipPath="url(#clip0_1_5)">
      <path d="M8 7H16" stroke="url(#paint0_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M5 12H19" stroke="url(#paint1_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M2 17H22" stroke="url(#paint2_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 2V22" stroke="url(#paint3_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 2L21 2" stroke="url(#paint4_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 7L21 7" stroke="url(#paint5_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 12L21 12" stroke="url(#paint6_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 17L21 17" stroke="url(#paint7_linear_1_5)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </g>
    <defs>
      <linearGradient id="paint0_linear_1_5" x1="5" y1="7" x2="19" y2="7" gradientUnits="userSpaceOnUse">
        <stop stopColor="#EC4899"/>
        <stop offset="1" stopColor="#F59E0B"/>
      </linearGradient>
      <linearGradient id="paint1_linear_1_5" x1="5" y1="12" x2="19" y2="12" gradientUnits="userSpaceOnUse">
        <stop stopColor="#06B6D4"/>
        <stop offset="1" stopColor="#3B82F6"/>
      </linearGradient>
      <linearGradient id="paint2_linear_1_5" x1="2" y1="17" x2="22" y2="17" gradientUnits="userSpaceOnUse">
        <stop stopColor="#8B5CF6"/>
        <stop offset="1" stopColor="#EC4899"/>
      </linearGradient>
      <linearGradient id="paint3_linear_1_5" x1="12" y1="2" x2="12" y2="22" gradientUnits="userSpaceOnUse">
        <stop stopColor="#F59E0B"/>
        <stop offset="1" stopColor="#FBBF24"/>
      </linearGradient>
      <linearGradient id="paint4_linear_1_5" x1="3" y1="2" x2="21" y2="2" gradientUnits="userSpaceOnUse">
        <stop stopColor="#EC4899"/>
        <stop offset="1" stopColor="#F43F5E"/>
      </linearGradient>
      <linearGradient id="paint5_linear_1_5" x1="3" y1="7" x2="21" y2="7" gradientUnits="userSpaceOnUse">
        <stop stopColor="#06B6D4"/>
        <stop offset="1" stopColor="#38BDF8"/>
      </linearGradient>
      <linearGradient id="paint6_linear_1_5" x1="3" y1="12" x2="21" y2="12" gradientUnits="userSpaceOnUse">
        <stop stopColor="#8B5CF6"/>
        <stop offset="1" stopColor="#A78BFA"/>
      </linearGradient>
      <linearGradient id="paint7_linear_1_5" x1="3" y1="17" x2="21" y2="17" gradientUnits="userSpaceOnUse">
        <stop stopColor="#F59E0B"/>
        <stop offset="1" stopColor="#FBBF24"/>
      </linearGradient>
      <clipPath id="clip0_1_5">
        <rect width="24" height="24" fill="white"/>
      </clipPath>
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
    Svg: RobotBrainSVG,
    description: (
      <>
        Explore cutting-edge concepts in artificial intelligence for robotics,
        from neural networks to machine learning algorithms that power autonomous machines.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    Svg: HumanoidRobotSVG,
    description: (
      <>
        Understand the mechanics and control systems behind humanoid robots,
        including gait planning, balance control, and human-robot interaction.
      </>
    ),
  },
  {
    title: 'Interactive Labs',
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
