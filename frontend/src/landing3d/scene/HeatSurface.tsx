import { useFrame } from '@react-three/fiber';
import { useMemo, useRef, useEffect } from 'react';
import type { MutableRefObject } from 'react';
import { BufferAttribute, BufferGeometry, Color, Mesh, MathUtils } from 'three';
import type { QualitySettings } from '../lib/perf/quality';
import { getThemeColors } from '../lib/useThemeColors';

interface HeatSurfaceProps {
  width?: number;
  depth?: number;
  segments?: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
  colorblindMode?: boolean;
  customColors?: { up: string; down: string };
}

// Seeded random for deterministic patterns
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 12.9898 + seed * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

export function HeatSurface({
  width = 16,
  depth = 10,
  segments = 40,
  visibilityRef,
  quality,
  colorblindMode = false,
  customColors,
}: HeatSurfaceProps) {
  const meshRef = useRef<Mesh>(null);
  const geometryRef = useRef<BufferGeometry>(null);
  
  const colors = useMemo(
    () => getThemeColors(colorblindMode, customColors),
    [colorblindMode, customColors]
  );

  // Reduce segments for lower quality
  const actualSegments = quality.tier === 'low' ? Math.floor(segments * 0.6) : segments;

  // Generate base height map (represents sector performance)
  const heightMap = useMemo(() => {
    const map: number[][] = [];
    for (let i = 0; i <= actualSegments; i++) {
      map[i] = [];
      for (let j = 0; j <= actualSegments; j++) {
        // Create some "hot spots" and "cold spots"
        const x = i / actualSegments;
        const z = j / actualSegments;
        
        // Multiple sine waves for interesting terrain
        const wave1 = Math.sin(x * Math.PI * 2) * Math.cos(z * Math.PI * 3) * 0.3;
        const wave2 = Math.sin(x * Math.PI * 4 + 1) * Math.sin(z * Math.PI * 2) * 0.2;
        const noise = (seededRandom(i * 100 + j) - 0.5) * 0.1;
        
        map[i][j] = wave1 + wave2 + noise;
      }
    }
    return map;
  }, [actualSegments]);

  // Initialize geometry
  useEffect(() => {
    if (!geometryRef.current) return;

    const geo = geometryRef.current;
    const positionAttr = geo.attributes.position as BufferAttribute;
    const colorAttr = geo.attributes.color as BufferAttribute;

    if (!positionAttr || !colorAttr) return;

    const tempColor = new Color();

    for (let i = 0; i <= actualSegments; i++) {
      for (let j = 0; j <= actualSegments; j++) {
        const idx = i * (actualSegments + 1) + j;
        const height = heightMap[i][j];

        // Set position
        const x = (i / actualSegments - 0.5) * width;
        const z = (j / actualSegments - 0.5) * depth;
        positionAttr.setXYZ(idx, x, height, z);

        // Set color based on height
        if (height >= 0) {
          tempColor.lerpColors(new Color('#1a1a2e'), colors.positiveColor, Math.min(height * 3, 1));
        } else {
          tempColor.lerpColors(new Color('#1a1a2e'), colors.negativeColor, Math.min(-height * 3, 1));
        }
        colorAttr.setXYZ(idx, tempColor.r, tempColor.g, tempColor.b);
      }
    }

    positionAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
    geo.computeVertexNormals();
  }, [heightMap, actualSegments, width, depth, colors]);

  useFrame((state) => {
    if (!visibilityRef.current || !meshRef.current || !geometryRef.current) return;

    const time = state.clock.elapsedTime;
    const geo = geometryRef.current;
    const positionAttr = geo.attributes.position as BufferAttribute;
    const colorAttr = geo.attributes.color as BufferAttribute;

    if (!positionAttr || !colorAttr) return;

    const tempColor = new Color();
    const waveSpeed = 0.5 * quality.motionScale;

    for (let i = 0; i <= actualSegments; i++) {
      for (let j = 0; j <= actualSegments; j++) {
        const idx = i * (actualSegments + 1) + j;
        const baseHeight = heightMap[i][j];

        // Animated wave overlay
        const x = i / actualSegments;
        const z = j / actualSegments;
        const wave = Math.sin(x * Math.PI * 2 + time * waveSpeed) * 
                     Math.cos(z * Math.PI * 2 - time * waveSpeed * 0.7) * 0.1;

        const height = baseHeight + wave;
        positionAttr.setY(idx, height);

        // Update color with animation
        const intensity = Math.abs(height) * 2 + Math.sin(time + i * 0.1) * 0.1;
        if (height >= 0) {
          tempColor.lerpColors(new Color('#1a1a2e'), colors.positiveColor, MathUtils.clamp(intensity, 0, 1));
        } else {
          tempColor.lerpColors(new Color('#1a1a2e'), colors.negativeColor, MathUtils.clamp(intensity, 0, 1));
        }
        colorAttr.setXYZ(idx, tempColor.r, tempColor.g, tempColor.b);
      }
    }

    positionAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
    geo.computeVertexNormals();
  });

  // Create geometry with position and color attributes
  const geometry = useMemo(() => {
    const geo = new BufferGeometry();
    const vertices: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];

    // Create vertices
    for (let i = 0; i <= actualSegments; i++) {
      for (let j = 0; j <= actualSegments; j++) {
        const x = (i / actualSegments - 0.5) * width;
        const z = (j / actualSegments - 0.5) * depth;
        vertices.push(x, 0, z);
        colors.push(0.1, 0.1, 0.15); // Initial color
      }
    }

    // Create indices for triangles
    for (let i = 0; i < actualSegments; i++) {
      for (let j = 0; j < actualSegments; j++) {
        const a = i * (actualSegments + 1) + j;
        const b = a + 1;
        const c = a + actualSegments + 1;
        const d = c + 1;

        indices.push(a, c, b);
        indices.push(b, c, d);
      }
    }

    geo.setIndex(indices);
    geo.setAttribute('position', new BufferAttribute(new Float32Array(vertices), 3));
    geo.setAttribute('color', new BufferAttribute(new Float32Array(colors), 3));
    geo.computeVertexNormals();

    return geo;
  }, [actualSegments, width, depth]);

  return (
    <mesh ref={meshRef} rotation={[0, 0, 0]} position={[0, -1, 0]}>
      <primitive object={geometry} ref={geometryRef} attach="geometry" />
      <meshStandardMaterial
        vertexColors
        metalness={0.3}
        roughness={0.7}
        transparent
        opacity={0.85}
      />
    </mesh>
  );
}
