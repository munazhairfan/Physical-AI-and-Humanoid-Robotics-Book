import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import FloatingChat from '@site/src/components/ChatWidget/FloatingChat';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <FloatingChat />
    </>
  );
}