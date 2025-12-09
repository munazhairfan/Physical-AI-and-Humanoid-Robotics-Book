import React from 'react';
import styles from './AnimatedBlobs.module.css';

const AnimatedBlobs = () => {
  return (
    <div className={styles.blobContainer}>
      <div className={`${styles.blob} ${styles.blob1}`}></div>
      <div className={`${styles.blob} ${styles.blob2}`}></div>
      <div className={`${styles.blob} ${styles.blob3}`}></div>
      <div className={`${styles.blob} ${styles.blob4}`}></div>
    </div>
  );
};

export default AnimatedBlobs;