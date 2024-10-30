// app/components/Sidebar.tsx

'use client';

import React from 'react';
import styles from '../styles/Sidebar.module.css';

interface SidebarProps {
  position: 'left' | 'right';
  isActive: boolean;
  toggleSidebar: () => void;
  children: React.ReactNode;
}

const Sidebar: React.FC<SidebarProps> = ({ position, isActive, toggleSidebar, children }) => {
  return (
    <aside
      className={`${styles.sidebar} ${styles[position]} ${isActive ? styles.active : ''}`}
    >
      <button className={styles.closeButton} onClick={toggleSidebar}>
        &times;
      </button>
      {children}
    </aside>
  );
};

export default Sidebar;
