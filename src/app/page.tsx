"use client";

import React, { useRef, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import LayerToggle from "./components/LayerToggle";
import styles from "./styles/Home.module.css";
import { FiMenu } from "react-icons/fi";
import { Noise } from "noisejs";
import { Matrix, CholeskyDecomposition } from "ml-matrix";

interface ClickedPoint {
  x: number;
  y: number;
  value: number;
}

const HomePage: React.FC = () => {
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const interactionCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskDataCanvasRef = useRef<HTMLCanvasElement>(null);
  const trueCanvasRef = useRef<HTMLCanvasElement>(null);
  const gpMeanCanvasRef = useRef<HTMLCanvasElement>(null);
  const gpVarianceCanvasRef = useRef<HTMLCanvasElement>(null);
  const acquisitionCanvasRef = useRef<HTMLCanvasElement>(null);

  const [canvasSize, setCanvasSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [clickedPoints, setClickedPoints] = useState<ClickedPoint[]>([]);
  const [isLeftSidebarOpen, setLeftSidebarOpen] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [showFunctionOverlay, setShowFunctionOverlay] = useState(false);

  const [gpMeanData, setGpMeanData] = useState<number[][] | null>(null);
  const [gpVarianceData, setGpVarianceData] = useState<number[][] | null>(null);
  const [acquisitionData, setAcquisitionData] = useState<number[][] | null>(
    null
  );

  const [showGpMean, setShowGpMean] = useState(false);
  const [showGpVariance, setShowGpVariance] = useState(false);
  const [showAcquisition, setShowAcquisition] = useState(false);

  // ADD: State variable to store the random field data
  const [fieldData, setFieldData] = useState<number[][] | null>(null);

  const kernel = (x1, y1, x2, y2, lengthScale, sigmaF) => {
    const sqdist = (x1 - x2) ** 2 + (y1 - y2) ** 2;
    return sigmaF ** 2 * Math.exp((-0.5 * sqdist) / lengthScale ** 2);
  };

  const computeAcquisitionFunction = (gpMeanData, gpVarianceData, y_train) => {
    const acquisitionData = [];
    const yMax = Math.max(...y_train);
    const width = gpMeanData[0].length;
    const height = gpMeanData.length;

    for (let i = 0; i < height; i++) {
      const acqRow = [];
      for (let j = 0; j < width; j++) {
        const mu = gpMeanData[i][j];
        const sigma = Math.sqrt(Math.max(gpVarianceData[i][j], 1e-10)); // Avoid sqrt of negative number
        const z = (mu - yMax) / sigma;
        const ei = (mu - yMax) * cdf(z) + sigma * pdf(z);
        acqRow.push(ei);
      }
      acquisitionData.push(acqRow);
    }
    return acquisitionData;
  };

  // Helper functions for PDF and CDF of the standard normal distribution
  const pdf = (x) => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x);

  const cdf = (x) => {
    return (1 + erf(x / Math.sqrt(2))) / 2;
  };

  // Error function approximation
  const erf = (x) => {
    // Abramowitz and Stegun formula 7.1.26 approximation
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const t = 1 / (1 + p * x);
    const y =
      1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  };

  // GP regression useEffect
  useEffect(() => {
    if (!canvasSize || clickedPoints.length === 0) return;

    const computeGpPredictions = () => {
      if (!canvasSize || clickedPoints.length === 0) return;

      const lengthScale = 50; // Adjust as needed
      const sigmaF = 1.0;
      const noiseVariance = 1e-6;

      const X_train = clickedPoints.map((p) => [
        p.x * canvasSize.width,
        p.y * canvasSize.height,
      ]);
      const y_train = clickedPoints.map((p) => p.value);

      const width = canvasSize.width;
      const height = canvasSize.height;

      const stepSize = 20; // Adjust as needed
      const X_test = [];
      for (let i = 0; i < width; i += stepSize) {
        for (let j = 0; j < height; j += stepSize) {
          X_test.push([i, j]);
        }
      }

      // Compute the covariance matrix K between training points
      const K = Matrix.zeros(X_train.length, X_train.length);
      for (let i = 0; i < X_train.length; i++) {
        for (let j = 0; j < X_train.length; j++) {
          K.set(
            i,
            j,
            kernel(
              X_train[i][0],
              X_train[i][1],
              X_train[j][0],
              X_train[j][1],
              lengthScale,
              sigmaF
            )
          );
        }
      }

      // Add noise variance to the diagonal
      for (let i = 0; i < K.rows; i++) {
        K.set(i, i, K.get(i, i) + noiseVariance);
      }

      // Perform Cholesky decomposition on K
      let chol;
      try {
        chol = new CholeskyDecomposition(K);
      } catch (error) {
        console.error("Cholesky decomposition failed:", error);
        return;
      }
      const L = chol.lowerTriangularMatrix;

      // Compute the cross-covariance matrix K_s between training and test points
      const K_s = Matrix.zeros(X_train.length, X_test.length);
      for (let i = 0; i < X_train.length; i++) {
        for (let j = 0; j < X_test.length; j++) {
          K_s.set(
            i,
            j,
            kernel(
              X_train[i][0],
              X_train[i][1],
              X_test[j][0],
              X_test[j][1],
              lengthScale,
              sigmaF
            )
          );
        }
      }

      // Compute K_ss (diagonal only)
      const K_ss_diag = [];
      for (let i = 0; i < X_test.length; i++) {
        K_ss_diag.push(
          kernel(
            X_test[i][0],
            X_test[i][1],
            X_test[i][0],
            X_test[i][1],
            lengthScale,
            sigmaF
          )
        );
      }

      // Solve for alpha using Cholesky decomposition
      const y_trainMatrix = Matrix.columnVector(y_train);
      const alpha = chol.solve(y_trainMatrix); // (K + noise * I)^{-1} y_train

      // Compute predictive mean: gpMean = K_s^T * alpha
      const gpMeanVector = K_s.transpose().mmul(alpha); // Matrix of size [X_test.length x 1]

      // Compute predictive variance: gpVariance = K_ss - v^T * v
      const v = chol.solve(K_s); // v = (K + noise * I)^{-1} K_s
      const gpVarianceVector: number[] = [];
      for (let i = 0; i < X_test.length; i++) {
        const v_col = v.getColumn(i);
        const var_i =
          K_ss_diag[i] - v_col.reduce((sum, val) => sum + val * val, 0);
        gpVarianceVector.push(var_i);
      }

      // Reshape gpMean and gpVariance to 2D arrays matching downsampled grid
      const numX = Math.ceil(width / stepSize);
      const numY = Math.ceil(height / stepSize);

      const gpMeanData: number[][] = [];
      const gpVarianceData: number[][] = [];
      let idx = 0;
      for (let i = 0; i < numY; i++) {
        const meanRow: number[] = [];
        const varRow: number[] = [];
        for (let j = 0; j < numX; j++) {
          meanRow.push(gpMeanVector.get(idx, 0));
          varRow.push(gpVarianceVector[idx]);
          idx++;
        }
        gpMeanData.push(meanRow);
        gpVarianceData.push(varRow);
      }

      // Compute the acquisition function
      const acquisition = computeAcquisitionFunction(
        gpMeanData,
        gpVarianceData,
        y_train
      );

      // Update state
      setGpMeanData(gpMeanData);
      setGpVarianceData(gpVarianceData);
      setAcquisitionData(acquisition);
    };

    computeGpPredictions();
  }, [clickedPoints]);

  useEffect(() => {
    if (!canvasSize || !gpMeanData) return;

    const canvas = gpMeanCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!showGpMean) return;

    // Normalize gpMeanData
    const flattenedMean = gpMeanData.flat();
    const minMean = Math.min(...flattenedMean);
    const maxMean = Math.max(...flattenedMean);

    const numX = gpMeanData[0].length;
    const numY = gpMeanData.length;

    const cellWidth = canvas.width / numX;
    const cellHeight = canvas.height / numY;

    for (let i = 0; i < numY; i++) {
      for (let j = 0; j < numX; j++) {
        const normalizedValue =
          ((gpMeanData[i][j] - minMean) / (maxMean - minMean)) * 255;

        ctx.fillStyle = `rgba(${normalizedValue}, ${normalizedValue}, ${normalizedValue}, 0.6)`;
        ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
      }
    }
  }, [canvasSize, gpMeanData, showGpMean]);

  useEffect(() => {
    if (!canvasSize || !gpVarianceData) return;

    const canvas = gpVarianceCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!showGpVariance) return;
    
    // Normalize gpVarianceData
    const flattenedMean = gpVarianceData.flat();
    const minMean = Math.min(...flattenedMean);
    const maxMean = Math.max(...flattenedMean);

    const numX = gpVarianceData[0].length;
    const numY = gpVarianceData.length;

    const cellWidth = canvas.width / numX;
    const cellHeight = canvas.height / numY;

    for (let i = 0; i < numY; i++) {
      for (let j = 0; j < numX; j++) {
        const normalizedValue =
          ((gpVarianceData[i][j] - minMean) / (maxMean - minMean)) * 255;

        ctx.fillStyle = `rgba(${normalizedValue}, ${normalizedValue}, ${normalizedValue}, 0.6)`;
        ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
      }
    }
  }, [canvasSize, gpVarianceData, showGpVariance]);

  useEffect(() => {
    if (!canvasSize || !acquisitionData) return;

    const canvas = acquisitionCanvasRef.current;
    if (!canvas) return;

    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

     // Clear the canvas
     ctx.clearRect(0, 0, canvas.width, canvas.height);

     if (!showAcquisition) return;

    // Normalize acquisitionData to [0, 255] for visualization

    const flattenedAcq = acquisitionData.flat();
    const minMean = Math.min(...flattenedAcq);
    const maxMean = Math.max(...flattenedAcq);

    const numX = acquisitionData[0].length;
    const numY = acquisitionData.length;

    const cellWidth = canvas.width / numX;
    const cellHeight = canvas.height / numY;

    for (let i = 0; i < numY; i++) {
      for (let j = 0; j < numX; j++) {
        const normalizedValue =
          ((acquisitionData[i][j] - minMean) / (maxMean - minMean)) * 255;

        ctx.fillStyle = `rgba(${normalizedValue}, ${normalizedValue}, ${normalizedValue}, 0.6)`;
        ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
      }
    }
  }, [canvasSize, acquisitionData, showAcquisition]);

  // MODIFY: Remove the old trueFunction and replace it with the new one
  const trueFunction = (x: number, y: number) => {
    if (!fieldData || !canvasSize) return 0;

    // Convert coordinates to pixel indices
    const ix = Math.floor(x);
    const iy = Math.floor(y);

    if (ix >= 0 && ix < canvasSize.width && iy >= 0 && iy < canvasSize.height) {
      // Return the intensity from the field data
      return fieldData[iy][ix];
    } else {
      return 0;
    }
  };

  // Handle window resize for responsive canvas
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth * 0.9; // Adjusted for padding/margins
      const height = window.innerHeight * 0.9;
      const size = Math.min(width, height);
      setCanvasSize({ width: size, height: size });
    };

    window.addEventListener("resize", handleResize);
    handleResize(); // Initial size

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Load the map image and draw it on the map canvas
  useEffect(() => {
    if (!canvasSize) return; // Wait until canvasSize is set
    const mapCanvas = mapCanvasRef.current;
    if (!mapCanvas) return;
    const ctx = mapCanvas.getContext("2d");
    if (!ctx) return;

    const mapImg = new Image();
    mapImg.src = "/map.png";
    mapImg.onload = () => {
      ctx.drawImage(mapImg, 0, 0, mapCanvas.width, mapCanvas.height);
    };
  }, [canvasSize]);

  // Load the mask image and generate the function values
  useEffect(() => {
    if (!canvasSize) return; // Wait until canvasSize is set
    const loadMaskAndFunction = () => {
      const maskImg = new Image();
      maskImg.src = "/mask.png";
      maskImg.onload = () => {
        // Use the maskDataCanvasRef to store the off-screen canvas
        let maskDataCanvas = maskDataCanvasRef.current;
        if (!maskDataCanvas) {
          maskDataCanvas = document.createElement("canvas");
          maskDataCanvasRef.current = maskDataCanvas;
        }
        maskDataCanvas.width = canvasSize.width;
        maskDataCanvas.height = canvasSize.height;
        const ctx = maskDataCanvas.getContext("2d");
        if (!ctx) return;

        // Clear the canvas before drawing
        ctx.clearRect(0, 0, maskDataCanvas.width, maskDataCanvas.height);

        // Draw the mask image scaled to canvas size
        ctx.drawImage(
          maskImg,
          0,
          0,
          maskDataCanvas.width,
          maskDataCanvas.height
        );

        // Handle mask visibility
        const maskCanvasCtx = maskCanvasRef.current?.getContext("2d");
        if (maskCanvasCtx) {
          maskCanvasCtx.clearRect(0, 0, canvasSize.width, canvasSize.height);
          if (showMask) {
            maskCanvasCtx.drawImage(
              maskImg,
              0,
              0,
              canvasSize.width,
              canvasSize.height
            );
          }
        }
      };
      maskImg.onerror = () => {
        console.error("Failed to load mask image.");
      };
    };
    loadMaskAndFunction();
  }, [canvasSize, showMask]);

  const noise = useRef(new Noise(Math.random())).current;

  useEffect(() => {
    if (!canvasSize) return;

    // Check if the fieldData has already been generated
    if (fieldData) return;

    const width = canvasSize.width;
    const height = canvasSize.height;

    // Parameters for noise
    const baseScale = 0.0035;
    const octaves = 9; // Increase for more detail
    const persistence = 0.2;
    const lacunarity = 0.6;

    const noiseField = [];
    for (let y = 0; y < height; y++) {
      const row = [];
      for (let x = 0; x < width; x++) {
        let amplitude = 1;
        let frequency = baseScale;
        let noiseHeight = 0;

        for (let i = 0; i < octaves; i++) {
          const sampleX = x * frequency;
          const sampleY = y * frequency;

          const perlinValue = noise.perlin2(sampleX, sampleY) * 2 - 1;
          noiseHeight += perlinValue * amplitude;

          amplitude *= persistence;
          frequency *= lacunarity;
        }

        row.push(noiseHeight);
      }
      noiseField.push(row);
    }

    // Normalize the noise field to [0, 255]
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const val = noiseField[y][x];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
    const normalizedField = [];
    for (let y = 0; y < height; y++) {
      normalizedField[y] = [];
      for (let x = 0; x < width; x++) {
        const val = noiseField[y][x];
        // Normalize to [0, 255]
        const normalizedVal = ((val - minVal) / (maxVal - minVal)) * 255;
        normalizedField[y][x] = normalizedVal;
      }
    }

    // Set the field data
    setFieldData(normalizedField);
  }, [canvasSize, fieldData]);

  // Draw the true function overlay
  useEffect(() => {
    if (!canvasSize) return;
    // Access the canvas element from the ref
    const canvas = trueCanvasRef.current;
    if (!canvas) return;

    // Set the canvas dimensions
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;

    // Get the 2D rendering context
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear the entire canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Proceed only if the overlay should be shown
    if (showFunctionOverlay && fieldData) {
      // Create an ImageData object for pixel manipulation
      const imageData = ctx.createImageData(canvas.width, canvas.height);
      const data = imageData.data;

      // Iterate over each pixel in the canvas
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          // Calculate the index for the current pixel
          const index = (y * canvas.width + x) * 4;

          // Get the intensity value from fieldData
          const intensity = fieldData[y][x];

          // Ensure intensity is within [0, 255]
          const clampedIntensity = Math.max(0, Math.min(255, intensity));

          // Assign the intensity to RGB channels (grayscale)
          data[index] = clampedIntensity; // Red
          data[index + 1] = clampedIntensity; // Green
          data[index + 2] = clampedIntensity; // Blue
          data[index + 3] = 255; // Alpha (fully opaque)
        }
      }

      // Put the ImageData back onto the canvas
      ctx.putImageData(imageData, 0, 0);
    }
  }, [canvasSize, showFunctionOverlay, fieldData]);

  // Handle canvas clicks to add points and reveal function values
  const handleCanvasClick = (
    event:
      | React.MouseEvent<HTMLCanvasElement>
      | React.TouchEvent<HTMLCanvasElement>
  ) => {
    event.preventDefault();
    if (!canvasSize) return;
    const canvas = interactionCanvasRef.current;
    if (!canvas) {
      console.error("Interaction canvas not found.");
      return;
    }

    let clientX: number;
    let clientY: number;

    if ("touches" in event && event.touches.length > 0) {
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else if ("clientX" in event) {
      clientX = event.clientX;
      clientY = event.clientY;
    } else {
      console.warn("Unhandled event type.");
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = (clientX - rect.left) / rect.width;
    const y = (clientY - rect.top) / rect.height;

    console.log(
      `Canvas clicked at normalized coordinates: (${x.toFixed(2)}, ${y.toFixed(
        2
      )})`
    );

    // Convert normalized coordinates to canvas pixel coordinates and ensure integers
    const canvasX = Math.floor(x * canvasSize.width);
    const canvasY = Math.floor(y * canvasSize.height);

    // Access the mask data canvas
    const maskDataCanvas = maskDataCanvasRef.current;
    if (!maskDataCanvas) {
      console.error("Mask data canvas not found.");
      return;
    }
    const maskCtx = maskDataCanvas.getContext("2d");
    if (!maskCtx) {
      console.error("Failed to get context for mask data canvas.");
      return;
    }

    // Get the pixel data at the clicked location
    const pixelData = maskCtx.getImageData(canvasX, canvasY, 1, 1).data;
    const [r, g, b, a] = pixelData;

    // Determine if the pixel corresponds to ocean or land
    if (r < 128) {
      // Ocean (black pixel)
      console.log("Clicked on ocean");
      alert("You clicked on the ocean!");
      return; // Do not add the point
    } else {
      // Land (white pixel)
      console.log("Clicked on land");
      // Reveal function value

      const functionVal = trueFunction(x, y);

      // Add point to state
      setClickedPoints((prevPoints) => [
        ...prevPoints,
        { x, y, value: functionVal },
      ]);
    }
  };

  // Function to draw clicked points on the interaction canvas
  const drawClickedPoints = (
    ctx: CanvasRenderingContext2D,
    points: ClickedPoint[],
    width: number,
    height: number
  ) => {
    points.forEach((point) => {
      const canvasX = point.x * width;
      const canvasY = point.y * height;

      // Draw marker
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "blue";

      ctx.fill();
    });
  };

  // Draw clicked points on the interaction canvas
  useEffect(() => {
    if (!canvasSize) return;
    const canvas = interactionCanvasRef.current;
    if (!canvas) {
      console.error("Interaction canvas not found for drawing points.");
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.error("Failed to get context for interaction canvas.");
      return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawClickedPoints(ctx, clickedPoints, canvas.width, canvas.height);
    console.log("Clicked points drawn on interaction canvas.");
  }, [clickedPoints, canvasSize]);

  // Toggle functions for sidebars
  const toggleLeftSidebar = () => setLeftSidebarOpen(!isLeftSidebarOpen);

  const toggleMaskVisibility = () => {
    setShowMask(!showMask);
  };

  if (!canvasSize) {
    return null; // or a loading spinner if you prefer
  }

  return (
    <div className={styles.container}>
      <button className={styles.menuButtonLeft} onClick={toggleLeftSidebar}>
        <FiMenu />
      </button>

      <Sidebar
        position="left"
        isActive={isLeftSidebarOpen}
        toggleSidebar={toggleLeftSidebar}
      >
        <h2>Layers</h2>
        <LayerToggle
          label="Show Mask"
          checked={showMask}
          onChange={toggleMaskVisibility}
        />
        <LayerToggle
          label="Show Function Overlay"
          checked={showFunctionOverlay}
          onChange={() => setShowFunctionOverlay(!showFunctionOverlay)}
        />
        <LayerToggle
          label="Show GP Mean"
          checked={showGpMean}
          onChange={() => setShowGpMean(!showGpMean)}
        />
        <LayerToggle
          label="Show GP Variance"
          checked={showGpVariance}
          onChange={() => setShowGpVariance(!showGpVariance)}
        />
        <LayerToggle
          label="Show Acquisition Function"
          checked={showAcquisition}
          onChange={() => setShowAcquisition(!showAcquisition)}
        />

        <h2>Clicked Points</h2>
        {clickedPoints.map((point, index) => {
          return (
            <div key={index}>
              <p>x: {point.x.toFixed(2)}</p>
              <p>y: {point.y.toFixed(2)}</p>
              {/* Optionally display the function value at this point */}
              {fieldData && (
                <p>
                  Value:{" "}
                  {trueFunction(
                    point.x * canvasSize.width,
                    point.y * canvasSize.height
                  ).toFixed(2)}
                </p>
              )}
            </div>
          );
        })}
      </Sidebar>

      <main className={styles.main}>
        <div className={styles.canvasContainer}>
          <canvas
            ref={mapCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className={`${styles.canvas} ${styles.mapCanvas}`}
          />
          <canvas
            ref={maskCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className={`${styles.canvas} ${styles.maskCanvas}`}
          />
          <canvas
            ref={trueCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className={`${styles.canvas} ${styles.maskCanvas}`}
          />
          <canvas
            ref={gpMeanCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className={`${styles.canvas} ${styles.gpMeanCanvas}`}
          />
          <canvas
            ref={gpVarianceCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className={`${styles.canvas} ${styles.gpVarianceCanvas}`}
          />
          <canvas
            ref={acquisitionCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className={`${styles.canvas} ${styles.acquisitionCanvas}`}
          />

          <canvas
            ref={interactionCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            onClick={handleCanvasClick}
            onTouchStart={handleCanvasClick}
            className={`${styles.canvas} ${styles.interactionCanvas}`}
          />
        </div>
      </main>
    </div>
  );
};

export default HomePage;
