import { NormalizedLandmark } from '@mediapipe/tasks-vision';

export function flattenLandmarks(landmarks: NormalizedLandmark[]): number[] {
  if (!landmarks || landmarks.length === 0) return [];

  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];

  const hasCore = !!leftShoulder && !!rightShoulder && !!leftHip && !!rightHip;
  const anchorX = hasCore ? (leftHip.x + rightHip.x) * 0.5 : landmarks[0].x;
  const anchorY = hasCore ? (leftHip.y + rightHip.y) * 0.5 : landmarks[0].y;

  const shoulderMidX = hasCore ? (leftShoulder.x + rightShoulder.x) * 0.5 : anchorX;
  const shoulderMidY = hasCore ? (leftShoulder.y + rightShoulder.y) * 0.5 : anchorY;
  const torsoSize = Math.hypot(shoulderMidX - anchorX, shoulderMidY - anchorY);

  let maxSpan = 0;
  for (const kp of landmarks) {
    maxSpan = Math.max(maxSpan, Math.hypot(kp.x - anchorX, kp.y - anchorY));
  }

  const scale = Math.max(torsoSize, maxSpan * 0.4, 1e-4);
  const flat: number[] = [];
  for (const kp of landmarks) {
    flat.push((kp.x - anchorX) / scale);
    flat.push((kp.y - anchorY) / scale);
    flat.push((kp.z ?? 0) / scale);
  }

  // Add geometry features to improve discrimination between similar overall silhouettes.
  const keyIndices = [0, 11, 12, 13, 14, 15, 16, 23, 24];
  for (let i = 0; i < keyIndices.length; i++) {
    for (let j = i + 1; j < keyIndices.length; j++) {
      const a = landmarks[keyIndices[i]];
      const b = landmarks[keyIndices[j]];
      if (!a || !b) continue;
      flat.push(Math.hypot(a.x - b.x, a.y - b.y) / scale);
    }
  }

  return flat;
}

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(vecA.length, vecB.length);
  for (let i = 0; i < len; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function flattenHands(hands: NormalizedLandmark[][]): number[] {
  if (!hands || hands.length === 0) return [];

  // Keep a stable left-to-right order using wrist x to reduce frame-to-frame swapping.
  const sorted = [...hands]
    .filter((hand) => hand.length > 0)
    .sort((a, b) => (a[0]?.x ?? 0) - (b[0]?.x ?? 0))
    .slice(0, 2);

  const flat: number[] = [];
  for (const hand of sorted) {
    flat.push(...flattenSingleHand(hand));
  }
  return flat;
}

function flattenSingleHand(hand: NormalizedLandmark[]): number[] {
  if (!hand || hand.length === 0) return [];
  const wrist = hand[0] ?? hand[0];
  const middleMcp = hand[9] ?? hand[0];
  const scale = Math.max(
    Math.hypot((middleMcp?.x ?? wrist.x) - wrist.x, (middleMcp?.y ?? wrist.y) - wrist.y),
    1e-4
  );

  const flat: number[] = [];
  for (const kp of hand) {
    flat.push((kp.x - wrist.x) / scale);
    flat.push((kp.y - wrist.y) / scale);
    flat.push((kp.z ?? 0) / scale);
  }

  const fingerTips = [4, 8, 12, 16, 20];
  for (let i = 0; i < fingerTips.length; i++) {
    const tip = hand[fingerTips[i]];
    if (!tip) continue;
    flat.push(Math.hypot(tip.x - wrist.x, tip.y - wrist.y) / scale);
  }
  return flat;
}

function clippedCosineSimilarity(vecA: number[], vecB: number[]): number {
  return Math.max(0, cosineSimilarity(vecA, vecB));
}

function distanceSimilarity(vecA: number[], vecB: number[]): number {
  const len = Math.min(vecA.length, vecB.length);
  if (len === 0) return 0;
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const d = vecA[i] - vecB[i];
    sum += d * d;
  }
  const rms = Math.sqrt(sum / len);
  return 1 / (1 + rms * 3.5);
}

function robustSimilarity(vecA: number[], vecB: number[]): number {
  const c = clippedCosineSimilarity(vecA, vecB);
  const d = distanceSimilarity(vecA, vecB);
  return c * 0.55 + d * 0.45;
}

type CombinedSimilarityInput = {
  poseA: number[] | null;
  poseB: number[] | null;
  handsA: number[];
  handsB: number[];
};

export function combinedSimilarity(input: CombinedSimilarityInput): number {
  const hasPose = !!input.poseA && !!input.poseB;
  const hasHands = input.handsA.length > 0 && input.handsB.length > 0;

  if (!hasPose && !hasHands) return 0;
  if (hasPose && !hasHands) return robustSimilarity(input.poseA!, input.poseB!);
  if (!hasPose && hasHands) return robustSimilarity(input.handsA, input.handsB) * 0.82;

  const poseScore = robustSimilarity(input.poseA!, input.poseB!);
  const handScore = robustSimilarity(input.handsA, input.handsB);
  return poseScore * 0.72 + handScore * 0.28;
}
