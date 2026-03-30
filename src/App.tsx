import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  NormalizedLandmark,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';
import { combinedSimilarity, flattenHands, flattenLandmarks } from './lib/matcher';

type AppMetadata = {
  memeImages?: string[];
  defaultImage?: string;
};

type Meme = {
  id: string;
  url: string;
  poseVector: number[] | null;
  handVector: number[];
};

type CalibrationSample = {
  poseVector: number[] | null;
  handVector: number[];
};

type LiveScore = {
  id: string;
  label: string;
  score: number;
};

const FALLBACK_MEMES = [
  '/1.jpg',
  '/2.jpg',
  '/3.jpg',
  '/4.jpg',
  '/5.jpg',
  '/6.jpg',
  '/7.jpg',
  '/8.jpg',
];

const SCORE_UI_STRIDE = 6;
const FACE_STRIDE = 1;
const FACE_MISS_TOLERANCE = 20;
const FACE_SMOOTHING_ALPHA = 0.7;
const CALIBRATION_SAMPLES_PER_MEME = 50;
const CALIBRATION_FRAME_STRIDE = 2;
const CALIBRATION_MIN_SIGNAL = 0.55;
const CALIBRATION_MAX_SIMILARITY = 0.992;
const CALIBRATION_MEME_DELAY_MS = 3000;
const IMPORTANT_POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24];

function normalizeAssetPath(path: string): string {
  const withSlash = path.startsWith('/') ? path : `/${path}`;
  return encodeURI(withSlash);
}

function fileLabel(url: string): string {
  const raw = url.split('/').pop() || url;
  return decodeURIComponent(raw);
}

function mirrorLandmarks(landmarks: NormalizedLandmark[]): NormalizedLandmark[] {
  return landmarks.map((kp) => ({ ...kp, x: 1 - kp.x }));
}

function mirrorHands(hands: NormalizedLandmark[][]): NormalizedLandmark[][] {
  return hands.map((hand) => mirrorLandmarks(hand));
}

function smoothLandmarks(
  previous: NormalizedLandmark[] | null,
  current: NormalizedLandmark[],
  alpha: number
): NormalizedLandmark[] {
  if (!previous || previous.length !== current.length) {
    return current;
  }
  const oneMinusAlpha = 1 - alpha;
  return current.map((kp, idx) => {
    const prev = previous[idx];
    return {
      ...kp,
      x: prev.x * alpha + kp.x * oneMinusAlpha,
      y: prev.y * alpha + kp.y * oneMinusAlpha,
      z: (prev.z ?? 0) * alpha + (kp.z ?? 0) * oneMinusAlpha,
    };
  });
}

function poseVisibilityScore(landmarks: NormalizedLandmark[] | null): number {
  if (!landmarks || landmarks.length === 0) return 0;
  let total = 0;
  let count = 0;
  for (const idx of IMPORTANT_POSE_POINTS) {
    const kp = landmarks[idx];
    if (!kp) continue;
    total += kp.visibility ?? 0.5;
    count += 1;
  }
  return count > 0 ? total / count : 0;
}

export default function App() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isTraining, setIsTraining] = useState(true);

  const [poseLandmarkerVideo, setPoseLandmarkerVideo] = useState<PoseLandmarker | null>(null);
  const [poseLandmarkerImage, setPoseLandmarkerImage] = useState<PoseLandmarker | null>(null);
  const [handLandmarkerVideo, setHandLandmarkerVideo] = useState<HandLandmarker | null>(null);
  const [handLandmarkerImage, setHandLandmarkerImage] = useState<HandLandmarker | null>(null);
  const [faceLandmarkerVideo, setFaceLandmarkerVideo] = useState<FaceLandmarker | null>(null);

  const [memeUrls, setMemeUrls] = useState<string[]>([]);
  const [defaultImage, setDefaultImage] = useState('/default.jpg');
  const [memes, setMemes] = useState<Meme[]>([]);
  const [templateErrors, setTemplateErrors] = useState<string[]>([]);

  const [matchedMeme, setMatchedMeme] = useState<Meme | null>(null);
  const [liveScores, setLiveScores] = useState<LiveScore[]>([]);
  const [handsActive, setHandsActive] = useState(false);
  const [faceActive, setFaceActive] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationIndex, setCalibrationIndex] = useState(0);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationStatus, setCalibrationStatus] = useState('');
  const [sessionCalibrationSamples, setSessionCalibrationSamples] = useState<Record<string, CalibrationSample[]>>({});

  const matchedMemeRef = useRef<Meme | null>(null);
  const scoreFrameCounterRef = useRef(0);
  const handsActiveRef = useRef(false);
  const faceActiveRef = useRef(false);
  const previousFaceLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const recentFaceLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const missedFaceFramesRef = useRef(0);
  const calibrationBucketsRef = useRef<Record<string, CalibrationSample[]>>({});
  const calibrationStrideRef = useRef(0);
  const calibrationTransitionTimeRef = useRef(0);

  const sessionCalibrationCount = Object.values(sessionCalibrationSamples)
    .reduce((sum, list) => sum + list.length, 0);

  const clearTemporaryCalibration = () => {
    setSessionCalibrationSamples({});
    setCalibrationStatus('Temporary calibration cleared.');
  };

  const beginCalibration = () => {
    if (memes.length === 0 || isTraining) return;
    calibrationBucketsRef.current = {};
    calibrationStrideRef.current = 0;
    calibrationTransitionTimeRef.current = Date.now();
    setCalibrationIndex(0);
    setCalibrationProgress(0);
    setCalibrationStatus(`Hold pose for ${fileLabel(memes[0].url)} (${CALIBRATION_SAMPLES_PER_MEME} samples)`);
    setIsCalibrating(true);
  };

  const finishCalibration = (applySamples: boolean) => {
    if (!isCalibrating) return;
    if (applySamples) {
      const cloned: Record<string, CalibrationSample[]> = {};
      Object.entries(calibrationBucketsRef.current).forEach(([key, list]) => {
        cloned[key] = list.map((s) => ({
          poseVector: s.poseVector ? [...s.poseVector] : null,
          handVector: [...s.handVector],
        }));
      });
      setSessionCalibrationSamples(cloned);
      const total = Object.values(cloned).reduce((sum, list) => sum + list.length, 0);
      setCalibrationStatus(`Calibration complete. Added ${total} samples.`);
    } else {
      setCalibrationStatus('Calibration canceled.');
    }

    calibrationBucketsRef.current = {};
    calibrationStrideRef.current = 0;
    calibrationTransitionTimeRef.current = 0;
    setIsCalibrating(false);
    setCalibrationIndex(0);
    setCalibrationProgress(0);
    matchedMemeRef.current = null;
    setMatchedMeme(null);
  };

  useEffect(() => {
    const loadMetadata = async () => {
      try {
        const response = await fetch('/metadata.json', { cache: 'no-store' });
        if (!response.ok) throw new Error(`metadata fetch failed: ${response.status}`);
        const metadata = (await response.json()) as AppMetadata;

        const manifestMemes = (metadata.memeImages || [])
          .map((item) => item.trim())
          .filter(Boolean)
          .map(normalizeAssetPath);

        setMemeUrls(manifestMemes.length > 0 ? manifestMemes : FALLBACK_MEMES);

        if (metadata.defaultImage && metadata.defaultImage.trim()) {
          setDefaultImage(normalizeAssetPath(metadata.defaultImage.trim()));
        }
      } catch (error) {
        console.warn('Failed to load metadata. Using fallback meme list.', error);
        setMemeUrls(FALLBACK_MEMES);
      }
    };

    loadMetadata();
  }, []);

  useEffect(() => {
    let active = true;
    let createdPoseVideo: PoseLandmarker | null = null;
    let createdPoseImage: PoseLandmarker | null = null;
    let createdHandVideo: HandLandmarker | null = null;
    let createdHandImage: HandLandmarker | null = null;
    let createdFaceVideo: FaceLandmarker | null = null;

    const initModel = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm'
        );

        const baseOptions = { delegate: 'GPU' as const };

        const poseVideo = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            ...baseOptions,
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.3,
          minPosePresenceConfidence: 0.3,
          minTrackingConfidence: 0.3,
        });
        createdPoseVideo = poseVideo;

        const poseImage = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            ...baseOptions,
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
          },
          runningMode: 'IMAGE',
          numPoses: 1,
          minPoseDetectionConfidence: 0.1,
        });
        createdPoseImage = poseImage;

        const handVideo = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            ...baseOptions,
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.3,
          minHandPresenceConfidence: 0.3,
          minTrackingConfidence: 0.3,
        });
        createdHandVideo = handVideo;

        const handImage = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            ...baseOptions,
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          },
          runningMode: 'IMAGE',
          numHands: 2,
          minHandDetectionConfidence: 0.1,
        });
        createdHandImage = handImage;

        const faceVideo = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            ...baseOptions,
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: 0.3,
          minFacePresenceConfidence: 0.3,
          minTrackingConfidence: 0.3,
        });
        createdFaceVideo = faceVideo;

        if (!active) return;
        setPoseLandmarkerVideo(poseVideo);
        setPoseLandmarkerImage(poseImage);
        setHandLandmarkerVideo(handVideo);
        setHandLandmarkerImage(handImage);
        setFaceLandmarkerVideo(faceVideo);
        setIsModelLoaded(true);
      } catch (error) {
        console.error('Failed to initialize models:', error);
      }
    };

    initModel();

    return () => {
      active = false;
      if (createdPoseVideo) createdPoseVideo.close();
      if (createdPoseImage) createdPoseImage.close();
      if (createdHandVideo) createdHandVideo.close();
      if (createdHandImage) createdHandImage.close();
      if (createdFaceVideo) createdFaceVideo.close();
    };
  }, []);

  useEffect(() => {
    if (!poseLandmarkerImage || !handLandmarkerImage || memeUrls.length === 0) return;

    const trainTemplates = async () => {
      setIsTraining(true);
      const loaded: Meme[] = [];
      const errors: string[] = [];

      const offscreenCanvas = document.createElement('canvas');
      const offscreenCtx = offscreenCanvas.getContext('2d');

      for (const url of memeUrls) {
        try {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.src = url;

          await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Timeout loading image')), 5000);
            img.onload = () => {
              clearTimeout(timeout);
              resolve(null);
            };
            img.onerror = (e) => {
              clearTimeout(timeout);
              reject(e);
            };
          });

          if (!offscreenCtx) continue;
          offscreenCanvas.width = img.width;
          offscreenCanvas.height = img.height;
          offscreenCtx.clearRect(0, 0, img.width, img.height);
          offscreenCtx.drawImage(img, 0, 0);

          const poseResults = poseLandmarkerImage.detect(offscreenCanvas);
          const handResults = handLandmarkerImage.detect(offscreenCanvas);

          const hasPose = !!(poseResults.landmarks && poseResults.landmarks.length > 0);
          const hasHands = !!(handResults.landmarks && handResults.landmarks.length > 0);

          if (!hasPose && !hasHands) {
            errors.push(`No landmarks: ${fileLabel(url)}`);
            loaded.push({ id: url, url, poseVector: null, handVector: [] });
            continue;
          }

          const poseVector = hasPose ? flattenLandmarks(poseResults.landmarks[0]) : null;
          const handVector = hasHands ? flattenHands(handResults.landmarks || []) : [];

          loaded.push({
            id: url,
            url,
            poseVector,
            handVector,
          });
        } catch (error) {
          console.error(`Template load failed for ${url}:`, error);
          errors.push(`Failed: ${fileLabel(url)}`);
          loaded.push({ id: url, url, poseVector: null, handVector: [] });
        }
      }

      setMemes(loaded);
      setTemplateErrors(errors);
      setIsTraining(false);
    };

    trainTemplates();
  }, [poseLandmarkerImage, handLandmarkerImage, memeUrls]);

  useEffect(() => {
    if (!poseLandmarkerVideo || !handLandmarkerVideo || !faceLandmarkerVideo || !isModelLoaded || isTraining) {
      return;
    }

    let animationFrameId: number;
    let lastVideoTime = -1;

    const predict = () => {
      if (
        webcamRef.current &&
        webcamRef.current.video &&
        webcamRef.current.video.readyState === 4 &&
        canvasRef.current
      ) {
        const video = webcamRef.current.video;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (ctx) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            const startTimeMs = performance.now();

            const poseResults = poseLandmarkerVideo.detectForVideo(video, startTimeMs);
            const handResults = handLandmarkerVideo.detectForVideo(video, startTimeMs);
            const shouldRunFace = scoreFrameCounterRef.current % FACE_STRIDE === 0;
            const faceResults = shouldRunFace ? faceLandmarkerVideo.detectForVideo(video, startTimeMs) : null;

            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const drawingUtils = new DrawingUtils(ctx);

            const hasPose = !!(poseResults.landmarks && poseResults.landmarks.length > 0);
            const hasHands = !!(handResults.landmarks && handResults.landmarks.length > 0);

            if (hasPose) {
              const pose = poseResults.landmarks[0];
              drawingUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, {
                color: '#00FF00',
                lineWidth: 4,
              });
              drawingUtils.drawLandmarks(pose, {
                color: '#FF0000',
                lineWidth: 2,
                radius: 4,
              });
            }

            if (hasHands && handResults.landmarks) {
              for (const hand of handResults.landmarks) {
                drawingUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
                  color: '#22d3ee',
                  lineWidth: 2,
                });
                drawingUtils.drawLandmarks(hand, {
                  color: '#fde047',
                  lineWidth: 1,
                  radius: 3,
                });
              }
            }

            if (faceResults?.faceLandmarks && faceResults.faceLandmarks.length > 0) {
              const rawFace = faceResults.faceLandmarks[0];
              const smoothFace = smoothLandmarks(previousFaceLandmarksRef.current, rawFace, FACE_SMOOTHING_ALPHA);
              previousFaceLandmarksRef.current = smoothFace;
              recentFaceLandmarksRef.current = smoothFace;
              missedFaceFramesRef.current = 0;
              drawingUtils.drawLandmarks(smoothFace, {
                color: '#ef4444',
                lineWidth: 1,
                radius: 1.2,
              });
              if (!faceActiveRef.current) {
                faceActiveRef.current = true;
                setFaceActive(true);
              }
            } else if (shouldRunFace && recentFaceLandmarksRef.current && missedFaceFramesRef.current < FACE_MISS_TOLERANCE) {
              missedFaceFramesRef.current += 1;
              drawingUtils.drawLandmarks(recentFaceLandmarksRef.current, {
                color: '#ef4444',
                lineWidth: 1,
                radius: 1.2,
              });
            } else if (!shouldRunFace && recentFaceLandmarksRef.current) {
              drawingUtils.drawLandmarks(recentFaceLandmarksRef.current, {
                color: '#ef4444',
                lineWidth: 1,
                radius: 1.2,
              });
            } else if (faceActiveRef.current && shouldRunFace) {
              recentFaceLandmarksRef.current = null;
              previousFaceLandmarksRef.current = null;
              missedFaceFramesRef.current = 0;
              faceActiveRef.current = false;
              setFaceActive(false);
            }

            if (handsActiveRef.current !== hasHands) {
              handsActiveRef.current = hasHands;
              setHandsActive(hasHands);
            }

            if (memes.length > 0) {
              const currentPoseLandmarks = hasPose ? poseResults.landmarks[0] : null;
              const currentHandLandmarks = hasHands ? (handResults.landmarks || []) : [];

              const mirroredPose = currentPoseLandmarks ? mirrorLandmarks(currentPoseLandmarks) : null;
              const mirroredHands = currentHandLandmarks.length > 0 ? mirrorHands(currentHandLandmarks) : [];

              const currentPoseVector = mirroredPose ? flattenLandmarks(mirroredPose) : null;
              const currentHandVector = mirroredHands.length > 0 ? flattenHands(mirroredHands) : [];

              // Calculate signal quality for both calibration and matching
              const poseQuality = poseVisibilityScore(currentPoseLandmarks);
              const handQuality = hasHands ? Math.min(1, (handResults.landmarks?.length || 0) / 2) : 0;
              const signalQuality = poseQuality * 0.75 + handQuality * 0.25;
              const MINIMUM_MATCH_QUALITY = 0.45;

              if (isCalibrating) {
                const targetMeme = memes[calibrationIndex] || null;
                if (targetMeme) {
                  if (matchedMemeRef.current?.id !== targetMeme.id) {
                    matchedMemeRef.current = targetMeme;
                    setMatchedMeme(targetMeme);
                  }

                  const poseQuality = poseVisibilityScore(currentPoseLandmarks);
                  const handQuality = hasHands ? Math.min(1, (handResults.landmarks?.length || 0) / 2) : 0;
                  const signalQuality = poseQuality * 0.75 + handQuality * 0.25;
                  const hasFeature = !!currentPoseVector || currentHandVector.length > 0;

                  if (hasFeature && signalQuality >= CALIBRATION_MIN_SIGNAL) {
                    calibrationStrideRef.current += 1;
                    if (calibrationStrideRef.current % CALIBRATION_FRAME_STRIDE === 0) {
                      const sample: CalibrationSample = {
                        poseVector: currentPoseVector ? [...currentPoseVector] : null,
                        handVector: [...currentHandVector],
                      };
                      const bucket = calibrationBucketsRef.current[targetMeme.id] || [];
                      const lastSample = bucket[bucket.length - 1];
                      const isDiverse = !lastSample || combinedSimilarity({
                        poseA: sample.poseVector,
                        poseB: lastSample.poseVector,
                        handsA: sample.handVector,
                        handsB: lastSample.handVector,
                      }) < CALIBRATION_MAX_SIMILARITY;

                      if (isDiverse) {
                        bucket.push(sample);
                        calibrationBucketsRef.current[targetMeme.id] = bucket;
                      }

                      setCalibrationProgress(Math.min(1, bucket.length / CALIBRATION_SAMPLES_PER_MEME));
                      setCalibrationStatus(`Calibrating ${fileLabel(targetMeme.url)} (${bucket.length}/${CALIBRATION_SAMPLES_PER_MEME})`);

                      if (bucket.length >= CALIBRATION_SAMPLES_PER_MEME) {
                        const now = Date.now();
                        const timeSinceReady = now - calibrationTransitionTimeRef.current;

                        if (timeSinceReady < CALIBRATION_MEME_DELAY_MS) {
                          // Still waiting for delay to pass
                          const remainingMs = Math.max(0, CALIBRATION_MEME_DELAY_MS - timeSinceReady);
                          const remainingSec = Math.ceil(remainingMs / 1000);
                          const nextIndex = calibrationIndex + 1;
                          if (nextIndex < memes.length) {
                            setCalibrationStatus(`Done! ✓ Get ready for ${fileLabel(memes[nextIndex].url)} in ${remainingSec}s...`);
                          } else {
                            setCalibrationStatus(`Done with all poses! Finishing calibration in ${remainingSec}s...`);
                          }
                        } else {
                          // Delay has passed, transition to next meme
                          const nextIndex = calibrationIndex + 1;
                          if (nextIndex < memes.length) {
                            calibrationTransitionTimeRef.current = Date.now();
                            setCalibrationIndex(nextIndex);
                            setCalibrationProgress(0);
                            setCalibrationStatus(`Hold pose for ${fileLabel(memes[nextIndex].url)} (${CALIBRATION_SAMPLES_PER_MEME} samples)`);
                            calibrationStrideRef.current = 0;
                          } else {
                            finishCalibration(true);
                          }
                        }
                      } else if (calibrationTransitionTimeRef.current === 0) {
                        // Initialize transition time when we start (first meme)
                        calibrationTransitionTimeRef.current = Date.now();
                      }
                    }
                  } else {
                    setCalibrationStatus(`Need clearer pose/hands for ${fileLabel(targetMeme.url)} (${Math.round(signalQuality * 100)}%)`);
                  }
                }

                ctx.restore();
                animationFrameId = requestAnimationFrame(predict);
                return;
              }

              // Only perform matching if signal quality is sufficient
              if (signalQuality >= MINIMUM_MATCH_QUALITY) {
                const scoresById = new Map<string, number>();
                let bestMatch: Meme | null = null;
                let highestScore = -1;

                for (const meme of memes) {
                  let bestScore = combinedSimilarity({
                    poseA: currentPoseVector,
                    poseB: meme.poseVector,
                    handsA: currentHandVector,
                    handsB: meme.handVector,
                  });

                  const calibrated = sessionCalibrationSamples[meme.id] || [];
                  for (const sample of calibrated) {
                    const sampleScore = combinedSimilarity({
                    poseA: currentPoseVector,
                    poseB: sample.poseVector,
                    handsA: currentHandVector,
                    handsB: sample.handVector,
                  });
                  if (sampleScore > bestScore) bestScore = sampleScore;
                }

                scoresById.set(meme.id, bestScore);
                if (bestScore > highestScore) {
                  highestScore = bestScore;
                  bestMatch = meme;
                }
              }

              scoreFrameCounterRef.current += 1;
              if (scoreFrameCounterRef.current % SCORE_UI_STRIDE === 0) {
                const top = Array.from(scoresById.entries())
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 5)
                  .map(([id, score]) => ({ id, label: fileLabel(id), score }));
                setLiveScores(top);
              }

              const current = matchedMemeRef.current;
              const currentScore = current ? (scoresById.get(current.id) ?? -1) : -1;
              const SWITCH_MARGIN = 0.05;
              const ENTER_THRESHOLD = 0.5;
              const EXIT_THRESHOLD = 0.4;

              if (!current) {
                if (bestMatch && highestScore >= ENTER_THRESHOLD) {
                  matchedMemeRef.current = bestMatch;
                  setMatchedMeme(bestMatch);
                } else {
                  matchedMemeRef.current = null;
                  setMatchedMeme(null);
                }
              } else {
                if (currentScore < EXIT_THRESHOLD) {
                  if (bestMatch && highestScore >= ENTER_THRESHOLD) {
                    matchedMemeRef.current = bestMatch;
                    setMatchedMeme(bestMatch);
                  } else {
                    matchedMemeRef.current = null;
                    setMatchedMeme(null);
                  }
                } else if (bestMatch && bestMatch.id !== current.id && highestScore - currentScore >= SWITCH_MARGIN) {
                  matchedMemeRef.current = bestMatch;
                  setMatchedMeme(bestMatch);
                }
              }
              } else {
                // Quality too low - clear any matched meme and show default
                matchedMemeRef.current = null;
                setMatchedMeme(null);
                setLiveScores([]);
              }
            }

            ctx.restore();
          }
        }
      }

      animationFrameId = requestAnimationFrame(predict);
    };

    predict();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [
    poseLandmarkerVideo,
    handLandmarkerVideo,
    faceLandmarkerVideo,
    isModelLoaded,
    isTraining,
    memes,
    isCalibrating,
    calibrationIndex,
    sessionCalibrationSamples,
  ]);

  return (
    <div className="h-screen w-screen bg-black flex overflow-hidden">
      <div className="w-1/2 h-full relative border-r border-neutral-800">
        <Webcam
          ref={webcamRef}
          audio={false}
          mirrored={true}
          className="w-full h-full object-cover"
          videoConstraints={{ facingMode: 'user' }}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
          style={{ transform: 'scaleX(-1)' }}
        />

        <div className="absolute top-4 left-4 bg-black/70 text-white text-[10px] font-mono p-3 rounded-lg z-10 border border-white/10">
          <div className="mb-2 font-bold border-b border-white/20 pb-1">SYSTEM STATUS</div>
          <div>Model: <span className={isModelLoaded ? 'text-green-400' : 'text-yellow-400'}>{isModelLoaded ? 'READY' : 'LOADING...'}</span></div>
          <div>Memes: <span className={isTraining ? 'text-yellow-400' : 'text-green-400'}>{isTraining ? 'TRAINING...' : `${memes.length} LOADED`}</span></div>
          <div>Hands: <span className={handsActive ? 'text-green-400' : 'text-neutral-400'}>{handsActive ? 'ACTIVE' : 'INACTIVE'}</span></div>
          <div>Face: <span className={faceActive ? 'text-green-400' : 'text-neutral-400'}>{faceActive ? 'ACTIVE' : 'INACTIVE'}</span></div>

          <div className="mt-2 text-[11px] font-semibold tracking-wide text-white">LIVE SCORES</div>
          {liveScores.length > 0 ? (
            liveScores.map((item) => (
              <div key={item.id} className="flex justify-between gap-3">
                <span>{item.label}</span>
                <span className="text-emerald-400">{item.score.toFixed(3)}</span>
              </div>
            ))
          ) : (
            <div className="text-neutral-400">No confident match</div>
          )}

          <div className="mt-2 border border-emerald-700 rounded px-2 py-1 text-center text-emerald-300">
            MATCH: {matchedMeme ? fileLabel(matchedMeme.url) : 'DEFAULT'}
          </div>

          {templateErrors.length > 0 && (
            <div className="mt-2 text-[9px] text-yellow-300">
              Templates skipped: {templateErrors.length}
              <div className="text-[8px] text-yellow-200 max-w-[220px] break-words">
                {templateErrors.slice(0, 3).join(' | ')}
              </div>
            </div>
          )}

          <div className="mt-2 border border-cyan-800 rounded p-2 bg-black/40">
            <div className="text-[10px] text-cyan-300">Calibration</div>
            <div className="text-[9px] text-cyan-200">
              {calibrationStatus || 'Capture live samples for each meme to improve matching.'}
            </div>
            {sessionCalibrationCount > 0 && !isCalibrating && (
              <div className="text-[9px] text-cyan-200 mt-1">
                Temporary samples: {sessionCalibrationCount}
              </div>
            )}
            {isCalibrating && (
              <div className="mt-1 h-1.5 w-full bg-cyan-950 rounded overflow-hidden">
                <div
                  className="h-full bg-cyan-400"
                  style={{ width: `${Math.round(calibrationProgress * 100)}%` }}
                />
              </div>
            )}
            <div className="mt-2 flex gap-1">
              {!isCalibrating ? (
                <>
                  <button
                    type="button"
                    onClick={beginCalibration}
                    disabled={isTraining || memes.length === 0}
                    className="px-2 py-1 rounded bg-cyan-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Calibrate Live
                  </button>
                  {sessionCalibrationCount > 0 && (
                    <button
                      type="button"
                      onClick={clearTemporaryCalibration}
                      className="px-2 py-1 rounded bg-slate-700 text-white"
                    >
                      Clear Temp
                    </button>
                  )}
                </>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => finishCalibration(true)}
                    className="px-2 py-1 rounded bg-emerald-700 text-white"
                  >
                    Finish
                  </button>
                  <button
                    type="button"
                    onClick={() => finishCalibration(false)}
                    className="px-2 py-1 rounded bg-red-700 text-white"
                  >
                    Cancel
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="w-1/2 h-full relative bg-neutral-900 flex items-center justify-center">
        <img
          src={matchedMeme ? matchedMeme.url : defaultImage}
          alt="Matched meme"
          className="w-full h-full object-cover"
          onError={(e) => {
            if (e.currentTarget.src.includes('default.jpg')) {
              e.currentTarget.src =
                'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
            }
          }}
        />
      </div>
    </div>
  );
}
