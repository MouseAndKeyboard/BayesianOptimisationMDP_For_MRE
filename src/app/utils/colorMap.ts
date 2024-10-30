// app/utils/colorMap.ts

export interface RGBColor {
    r: number;
    g: number;
    b: number;
  }
  
  /**
   * Maps a normalized value (0 to 1) to a color.
   * You can customize this function to use different color schemes.
   */
  export const valueToColor = (value: number): RGBColor => {
    // Simple viridis-like colormap
    const r = Math.floor(255 * value);
    const g = Math.floor(255 * (1 - value));
    const b = 128;
    return { r, g, b };
  };
  