"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as faceapi from "face-api.js";

/* ──────────────────────────────────────────────
   TYPES
   ────────────────────────────────────────────── */
type Props = {
  onCapture: (descriptor: number[]) => void;
  onCancel?: () => void;
  timeoutSeconds?: number;
};

type LivenessState =
  | "LOADING"
  | "WAITING"
  | "BLINK"
  | "VERIFYING"
  | "SUCCESS"
  | "FAILED";

/* ──────────────────────────────────────────────
   EYE BLINK DETECTION (EAR)
   ────────────────────────────────────────────── */
function calculateEAR(landmarks: faceapi.FaceLandmarks68) {
  const pts = landmarks.positions;

  // Helper function to calculate Euclidean distance
  const dist = (p1: faceapi.Point, p2: faceapi.Point) =>
    Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

  // Left eye (indices 36-41)
  const leftEyeEAR =
    (dist(pts[37], pts[41]) + dist(pts[38], pts[40])) / (2.0 * dist(pts[36], pts[39]));

  // Right eye (indices 42-47)
  const rightEyeEAR =
    (dist(pts[43], pts[47]) + dist(pts[44], pts[46])) / (2.0 * dist(pts[42], pts[45]));

  // Average EAR
  return (leftEyeEAR + rightEyeEAR) / 2.0;
}

/* ──────────────────────────────────────────────
   THRESHOLDS
   ────────────────────────────────────────────── */
const EAR_CLOSED_THRESHOLD = 0.20; // EAR below this is considered a blink

/* ──────────────────────────────────────────────
   INSTRUCTIONS MAP
   ────────────────────────────────────────────── */
const INSTRUCTIONS: Record<LivenessState, { text: string; emoji: string }> = {
  LOADING:   { text: "Loading face detection models…", emoji: "⏳" },
  WAITING:   { text: "Position your face in the frame", emoji: "🧑" },
  BLINK:     { text: "Please BLINK your eyes", emoji: "👁️" },
  VERIFYING: { text: "Liveness verified! Capturing face…", emoji: "📸" },
  SUCCESS:   { text: "Face captured successfully!", emoji: "✅" },
  FAILED:    { text: "Detection lost. Please try again.", emoji: "⚠️" },
};

/* ──────────────────────────────────────────────
   COMPONENT
   ────────────────────────────────────────────── */
export default function LivenessFaceCapture({
  onCapture,
  onCancel,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [state, setState] = useState<LivenessState>("LOADING");
  const [modelsReady, setModelsReady] = useState(false);
  const [ear, setEar] = useState(0);

  // Refs for state machine (used inside animation loop to avoid stale closures)
  const stateRef = useRef<LivenessState>("LOADING");
  const startTimeRef = useRef<number>(0);

  // Blink State tracking
  const hasClosedEyesRef = useRef<boolean>(false);
  const frameCountRef = useRef<number>(0);
  const lostFramesRef = useRef<number>(0);

  const updateState = useCallback((newState: LivenessState) => {
    stateRef.current = newState;
    setState(newState);
  }, []);

  /* ── Load Models + Start Camera ── */
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const MODEL_URL = "/models";
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);

        if (cancelled) return;
        setModelsReady(true);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Explicitly call play to handle mobile browser pause behavior
          videoRef.current.play().catch(e => console.error("Video play failed:", e));
        }

        updateState("WAITING");
        runDetectionLoop();
      } catch (err) {
        console.error("Init error:", err);
        updateState("FAILED");
      }
    }

    init();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [updateState]);

  /* ── Start Liveness Challenge ── */
  function startChallenge() {
    updateState("BLINK");
    startTimeRef.current = Date.now();
    hasClosedEyesRef.current = false;
  }

  /* ── Detection Loop ── */
  function runDetectionLoop() {
    async function detect() {
      const currentState = stateRef.current;

      // Stop conditions
      if (
        currentState === "VERIFYING" ||
        currentState === "SUCCESS" ||
        currentState === "FAILED" ||
        currentState === "LOADING"
      ) {
        return;
      }

      if (!videoRef.current || videoRef.current.readyState < 2) {
        animFrameRef.current = requestAnimationFrame(detect);
        return;
      }

      try {
        const detection = await faceapi
          .detectSingleFace(
            videoRef.current,
            new faceapi.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: 0.5 })
          )
          .withFaceLandmarks();

        if (detection) {
          lostFramesRef.current = 0;
          const currentEar = calculateEAR(detection.landmarks);
          
          frameCountRef.current++;
          if (frameCountRef.current % 5 === 0) {
            setEar(Number(currentEar.toFixed(2)));
          }

          // Draw overlay on canvas
          drawOverlay(detection, currentEar);

          // State machine transitions
          const cs = stateRef.current;

          if (cs === "BLINK") {
            if (currentEar < EAR_CLOSED_THRESHOLD) {
              hasClosedEyesRef.current = true;
            } else if (hasClosedEyesRef.current && currentEar >= EAR_CLOSED_THRESHOLD) {
              // Eyes opened again after being closed -> Blink complete!
              updateState("VERIFYING");
              hasClosedEyesRef.current = false;
              captureAndVerify();
              return;
            }
          }
        } else if (stateRef.current === "BLINK") {
          lostFramesRef.current++;
          if (lostFramesRef.current > 15) {
             hasClosedEyesRef.current = false;
             updateState("FAILED");
             return;
          }
        }
      } catch (err) {
        console.error("Detection error:", err);
      }

      animFrameRef.current = requestAnimationFrame(detect);
    }

    animFrameRef.current = requestAnimationFrame(detect);
  }

  /* ── Draw Face Overlay ── */
  function drawOverlay(
    detection: faceapi.WithFaceLandmarks<{ detection: faceapi.FaceDetection }>,
    earValue: number
  ) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 360;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw face bounding box
    const box = detection.detection.box;
    const cs = stateRef.current;
    const color = cs === "BLINK" ? "#3B82F6" : "#22C55E";

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw landmark dots
    ctx.fillStyle = color;
    detection.landmarks.positions.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
      ctx.fill();
    });

    // EAR text
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px monospace";
    ctx.shadowColor = "#000";
    ctx.shadowBlur = 4;
    ctx.fillText(`EAR: ${earValue.toFixed(2)}`, 10, 25);
    ctx.shadowBlur = 0;
  }

  /* ── Final Capture ── */
  async function captureAndVerify() {
    if (!videoRef.current) return;

    let detection = null;
    let attempts = 0;
    
    while (!detection && attempts < 5) {
      detection = await faceapi
        .detectSingleFace(
          videoRef.current,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 160 })
        )
        .withFaceLandmarks()
        .withFaceDescriptor();
      attempts++;
      if (!detection) {
        await new Promise(r => setTimeout(r, 100));
      }
    }

    if (!detection) {
      updateState("FAILED");
      return;
    }

    updateState("SUCCESS");
    const descriptor = Array.from(detection.descriptor);

    // Small delay so user sees the success state
    setTimeout(() => {
      onCapture(descriptor);
    }, 800);
  }

  /* ── Retry ── */
  function handleRetry() {
    setEar(0);
    hasClosedEyesRef.current = false;
    updateState("WAITING");
    runDetectionLoop();
  }

  /* ── UI ── */
  const info = INSTRUCTIONS[state];
  const isActive = state === "BLINK";

  // Progress indicator
  const stepsDone = state === "VERIFYING" || state === "SUCCESS" ? 1 : 0;

  return (
    <div className="space-y-4">
      {/* ── Instruction Banner ── */}
      <div
        className={`rounded-xl p-4 text-center transition-all duration-300 ${
          state === "FAILED"
            ? "bg-red-100 border-2 border-red-300 text-red-800"
            : state === "SUCCESS"
            ? "bg-green-100 border-2 border-green-300 text-green-800"
            : "bg-blue-50 border-2 border-blue-200 text-blue-800"
        }`}
      >
        <span className="text-2xl block mb-1">{info.emoji}</span>
        <p className="font-semibold text-base">{info.text}</p>

      </div>

      {/* ── Progress Steps ── */}
      {(isActive || state === "VERIFYING" || state === "SUCCESS") && (
        <div className="flex justify-center gap-2">
          <div
            className={`flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium transition-all ${
              stepsDone === 1
                ? "bg-green-500 text-white"
                : "bg-blue-500 text-white animate-pulse"
            }`}
          >
            {stepsDone === 1 ? "✓ Verified" : "Blink 1 Time"}
          </div>
        </div>
      )}

      {/* ── Camera Feed ── */}
      <div className="relative rounded-xl overflow-hidden border-2 border-gray-200 bg-black">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full rounded-xl"
          style={{ transform: "scaleX(-1)" }}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Live EAR Indicator */}
        {isActive && (
          <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded-lg font-mono">
            EAR: {ear.toFixed(2)}
          </div>
        )}
      </div>

      {/* ── Action Buttons ── */}
      <div className="flex gap-3">
        {state === "WAITING" && (
          <button
            onClick={startChallenge}
            disabled={!modelsReady}
            className={`flex-1 py-3 rounded-xl text-lg font-semibold transition-all ${
              modelsReady
                ? "bg-blue-600 text-white hover:bg-blue-700 active:scale-95"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }`}
          >
            {modelsReady ? "Start Liveness Check" : "Loading Models…"}
          </button>
        )}

        {state === "FAILED" && (
          <button
            onClick={handleRetry}
            className="flex-1 py-3 rounded-xl text-lg font-semibold bg-orange-500 text-white hover:bg-orange-600 active:scale-95 transition-all"
          >
            🔄 Retry
          </button>
        )}

        {onCancel && state !== "SUCCESS" && state !== "VERIFYING" && (
          <button
            onClick={onCancel}
            className="px-6 py-3 rounded-xl border border-gray-300 text-gray-700 hover:bg-gray-100 transition-all"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
