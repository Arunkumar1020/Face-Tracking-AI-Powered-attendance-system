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
  | "LOOK_LEFT"
  | "LOOK_RIGHT"
  | "LOOK_DOWN"
  | "VERIFYING"
  | "SUCCESS"
  | "TIMEOUT";

/* ──────────────────────────────────────────────
   HEAD POSE ESTIMATION (from 68 landmarks)
   
   Uses a simplified ear–nose–chin geometry:
     Yaw  → horizontal ratio of nose tip between left/right ear
     Pitch → vertical drop of nose tip relative to eye line
   ────────────────────────────────────────────── */
function estimateHeadPose(landmarks: faceapi.FaceLandmarks68) {
  const pts = landmarks.positions;

  // Key points (0-indexed from 68-landmark model)
  const noseTip = pts[30];    // tip of nose
  const leftEar = pts[0];     // left jaw edge (right side of screen)
  const rightEar = pts[16];   // right jaw edge (left side of screen)
  const leftEye = pts[36];    // left eye outer corner
  const rightEye = pts[45];   // right eye outer corner
  const chin = pts[8];        // bottom of chin

  // ── YAW (left-right) ──
  // Ratio of nose position between ear edges
  // 0.5 = facing straight, < 0.5 = looking right, > 0.5 = looking left
  const faceWidth = rightEar.x - leftEar.x;
  const noseOffset = noseTip.x - leftEar.x;
  const yawRatio = faceWidth !== 0 ? noseOffset / faceWidth : 0.5;

  // Convert to approximate degrees (-45 to +45)
  // Center at 0.5, positive = looking left, negative = looking right
  const yawDeg = (yawRatio - 0.5) * 90;

  // ── PITCH (up-down) ──
  // Ratio of nose-to-chin vs eye-to-chin distance
  const eyeCenterY = (leftEye.y + rightEye.y) / 2;
  const eyeToChin = chin.y - eyeCenterY;
  const noseToChin = chin.y - noseTip.y;
  const pitchRatio = eyeToChin !== 0 ? noseToChin / eyeToChin : 0.5;

  // Normal pitch ratio ~0.55-0.65. Lower = looking down, Higher = looking up
  // Convert to approximate degrees
  const pitchDeg = (pitchRatio - 0.6) * -100;

  return { yawDeg, pitchDeg };
}

/* ──────────────────────────────────────────────
   THRESHOLDS
   ────────────────────────────────────────────── */
const YAW_LEFT_THRESHOLD = 20;    // Yaw > 20° → looking left
const YAW_RIGHT_THRESHOLD = -20;  // Yaw < -20° → looking right
const PITCH_DOWN_THRESHOLD = -12; // Pitch < -12° → looking down
const DEFAULT_TIMEOUT = 10;       // seconds

/* ──────────────────────────────────────────────
   INSTRUCTIONS MAP
   ────────────────────────────────────────────── */
const INSTRUCTIONS: Record<LivenessState, { text: string; emoji: string }> = {
  LOADING:   { text: "Loading face detection models…", emoji: "⏳" },
  WAITING:   { text: "Position your face in the frame", emoji: "🧑" },
  LOOK_LEFT: { text: "Turn your head LEFT", emoji: "👈" },
  LOOK_RIGHT:{ text: "Now turn your head RIGHT", emoji: "👉" },
  LOOK_DOWN: { text: "Now look DOWN", emoji: "👇" },
  VERIFYING: { text: "Liveness verified! Capturing face…", emoji: "📸" },
  SUCCESS:   { text: "Face captured successfully!", emoji: "✅" },
  TIMEOUT:   { text: "Timed out. Please try again.", emoji: "⏰" },
};

/* ──────────────────────────────────────────────
   COMPONENT
   ────────────────────────────────────────────── */
export default function LivenessFaceCapture({
  onCapture,
  onCancel,
  timeoutSeconds = DEFAULT_TIMEOUT,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [state, setState] = useState<LivenessState>("LOADING");
  const [modelsReady, setModelsReady] = useState(false);
  const [yaw, setYaw] = useState(0);
  const [pitch, setPitch] = useState(0);

  // Refs for state machine (used inside animation loop to avoid stale closures)
  const stateRef = useRef<LivenessState>("LOADING");
  const startTimeRef = useRef<number>(0);

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
          video: { facingMode: "user", width: 480, height: 360 },
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        updateState("WAITING");
      } catch (err) {
        console.error("Init error:", err);
        updateState("TIMEOUT");
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
    updateState("LOOK_LEFT");
    startTimeRef.current = Date.now();
    runDetectionLoop();
  }

  /* ── Detection Loop ── */
  function runDetectionLoop() {
    async function detect() {
      const currentState = stateRef.current;

      // Stop conditions
      if (
        currentState === "VERIFYING" ||
        currentState === "SUCCESS" ||
        currentState === "TIMEOUT" ||
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
            new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 })
          )
          .withFaceLandmarks();

        if (detection) {
          const { yawDeg, pitchDeg } = estimateHeadPose(detection.landmarks);
          setYaw(Math.round(yawDeg));
          setPitch(Math.round(pitchDeg));

          // Draw overlay on canvas
          drawOverlay(detection, yawDeg, pitchDeg);

          // State machine transitions
          const cs = stateRef.current;

          if (cs === "LOOK_LEFT" && yawDeg > YAW_LEFT_THRESHOLD) {
            updateState("LOOK_RIGHT");
          } else if (cs === "LOOK_RIGHT" && yawDeg < YAW_RIGHT_THRESHOLD) {
            updateState("LOOK_DOWN");
          } else if (cs === "LOOK_DOWN" && pitchDeg < PITCH_DOWN_THRESHOLD) {
            updateState("VERIFYING");
            await captureAndVerify();
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
    yawDeg: number,
    pitchDeg: number
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
    const color =
      cs === "LOOK_LEFT" ? "#3B82F6" :
      cs === "LOOK_RIGHT" ? "#8B5CF6" :
      cs === "LOOK_DOWN" ? "#F59E0B" :
      "#22C55E";

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

    // Yaw/Pitch text
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px monospace";
    ctx.shadowColor = "#000";
    ctx.shadowBlur = 4;
    ctx.fillText(`Yaw: ${Math.round(yawDeg)}°`, 10, 25);
    ctx.fillText(`Pitch: ${Math.round(pitchDeg)}°`, 10, 45);
    ctx.shadowBlur = 0;
  }

  /* ── Final Capture ── */
  async function captureAndVerify() {
    if (!videoRef.current) return;

    const detection = await faceapi
      .detectSingleFace(
        videoRef.current,
        new faceapi.TinyFaceDetectorOptions()
      )
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      updateState("TIMEOUT");
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
    setYaw(0);
    setPitch(0);
    updateState("WAITING");
  }

  /* ── UI ── */
  const info = INSTRUCTIONS[state];
  const isActive = ["LOOK_LEFT", "LOOK_RIGHT", "LOOK_DOWN"].includes(state);

  // Progress indicator
  const stepsDone =
    state === "LOOK_LEFT" ? 0 :
    state === "LOOK_RIGHT" ? 1 :
    state === "LOOK_DOWN" ? 2 :
    state === "VERIFYING" || state === "SUCCESS" ? 3 : 0;

  return (
    <div className="space-y-4">
      {/* ── Instruction Banner ── */}
      <div
        className={`rounded-xl p-4 text-center transition-all duration-300 ${
          state === "TIMEOUT"
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
          {["Left", "Right", "Down"].map((label, i) => (
            <div
              key={label}
              className={`flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium transition-all ${
                i < stepsDone
                  ? "bg-green-500 text-white"
                  : i === stepsDone && isActive
                  ? "bg-blue-500 text-white animate-pulse"
                  : "bg-gray-200 text-gray-500"
              }`}
            >
              {i < stepsDone ? "✓" : i + 1} {label}
            </div>
          ))}
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

        {/* Live Yaw/Pitch Indicator */}
        {isActive && (
          <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded-lg font-mono">
            Y:{yaw}° P:{pitch}°
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

        {state === "TIMEOUT" && (
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
