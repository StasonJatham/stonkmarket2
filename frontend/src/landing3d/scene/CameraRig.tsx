import { useFrame, useThree } from '@react-three/fiber';
import { MathUtils, Vector3 } from 'three';
import { useScroll } from '@react-three/drei';
import { useMemo } from 'react';
import type { MutableRefObject } from 'react';
import type { QualitySettings } from '../lib/perf/quality';

interface CameraRigProps {
  positions: Vector3[];
  targets: Vector3[];
  quality: QualitySettings;
  visibilityRef: MutableRefObject<boolean>;
}

export function CameraRig({ positions, targets, quality, visibilityRef }: CameraRigProps) {
  const scroll = useScroll();
  const { camera } = useThree();
  const tempPosition = useMemo(() => new Vector3(), []);
  const tempTarget = useMemo(() => new Vector3(), []);

  useFrame(() => {
    if (!visibilityRef.current) return;

    const sectionSize = 1 / Math.max(1, positions.length - 1);
    const offset = MathUtils.clamp(scroll.offset, 0, 1);
    const scaled = offset / sectionSize;
    const index = Math.min(positions.length - 2, Math.floor(scaled));
    const localT = MathUtils.clamp(scaled - index, 0, 1);
    const eased = MathUtils.smootherstep(localT, 0, 1);

    tempPosition.lerpVectors(positions[index], positions[index + 1], eased);
    tempTarget.lerpVectors(targets[index], targets[index + 1], eased);

    const damping = quality.reducedMotion ? 0.12 : 0.08;

    camera.position.lerp(tempPosition, damping * quality.motionScale);
    camera.lookAt(tempTarget);
  });

  return null;
}
