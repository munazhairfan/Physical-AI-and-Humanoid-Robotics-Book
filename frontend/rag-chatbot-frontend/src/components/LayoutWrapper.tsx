import React, { ReactNode } from 'react';
import FloatingChat from '../ChatWidget/FloatingChat';

interface LayoutWrapperProps {
  children: ReactNode;
}

const LayoutWrapper: React.FC<LayoutWrapperProps> = ({ children }) => {
  return (
    <>
      {children}
      <FloatingChat />
    </>
  );
};

export default LayoutWrapper;