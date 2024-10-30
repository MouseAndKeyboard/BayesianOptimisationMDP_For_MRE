// app/components/LayerToggle.tsx

'use client';

import React from 'react';
import styles from '../styles/LayerToggle.module.css';

interface LayerToggleProps {
  label: string;
  checked: boolean;
  onChange: () => void;
}

const LayerToggle: React.FC<LayerToggleProps> = ({ label, checked, onChange }) => {
  return (
    <label className={styles.toggleLabel}>
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span className={styles.slider}></span>
      <span className={styles.labelText}>{label}</span>
    </label>
  );
};

export default LayerToggle;
