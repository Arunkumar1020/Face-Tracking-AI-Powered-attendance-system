"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

/* ──────────────────────────────────────────────
   TYPES
   ────────────────────────────────────────────── */
type Props = {
  // Updated from descriptor to base64 image for backend matching
  onCapture: (imageSrc: string) => void;
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

type ChallengeStep = "BLINK" | "TURN" | "DONE";

/* ──────────────────────────────────────────────
   INSTRUCTIONS MAP
   ────────────────────────────────────────────── */
const INSTRUCTIONS: Record<LivenessState, { text: string; emoji: string }> = {
  LOADING:   { text: "Loading face detection models…", emoji: "⏳" },
  WAITING:   { text: "Position your face in the frame", emoji: "🧑" },
  BLINK:     { text: "Liveness Check Active", emoji: "👁️" },
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
  const streamRef = useRef<MediaStream | null>(null);

  const [state, setState] = useState<LivenessState>("LOADING");
  const [modelsReady, setModelsReady] = useState(false);
  
  // Refs for performance optimizations
  const stateRef = useRef<LivenessState>("LOADING");
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);

  // State machine for liveness challenge
  const challengeStepRef = useRef<ChallengeStep>("BLINK");
  const lostFramesRef = useRef<number>(0);
  const liveTextRef = useRef<HTMLDivElement>(null);

  const updateState = useCallback((newState: LivenessState) => {
    stateRef.current = newState;
    setState(newState);
  }, []);

  const updateLiveText = (text: string) => {
    if (liveTextRef.current && liveTextRef.current.innerText !== text) {
      liveTextRef.current.innerText = text;
    }
  };

  /* ── Load Models + Start Camera ── */
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU" // Closest mapping for IMAGE_FAST high-speed hardware acc
          },
          runningMode: "VIDEO",
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
          numFaces: 1
        });

        if (cancelled) {
          faceLandmarker.close();
          return;
        }

        faceLandmarkerRef.current = faceLandmarker;
        setModelsReady(true);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(e => console.error("Video play failed:", e));
        }

        updateState("WAITING");
        
        // Wait for video to be ready before starting loop
        videoRef.current?.addEventListener('loadeddata', () => {
           if (!cancelled) runDetectionLoop();
        });

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
      if (faceLandmarkerRef.current) {
        faceLandmarkerRef.current.close();
      }
    };
  }, [updateState]);

  /* ── Start Liveness Challenge ── */
  function startChallenge() {
    challengeStepRef.current = "BLINK";
    updateState("BLINK");
    updateLiveText("Please BLINK your eyes");
    lostFramesRef.current = 0;
  }

  /* ── 60 FPS Detection Loop ── */
  function runDetectionLoop() {
    const detect = () => {
      const cs = stateRef.current;
      if (cs === "SUCCESS" || cs === "FAILED" || cs === "VERIFYING" || cs === "LOADING") return;

      if (videoRef.current && videoRef.current.readyState >= 2 && faceLandmarkerRef.current) {
        const video = videoRef.current;
        const currentTime = video.currentTime;
        
        if (currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = currentTime;
          // Using video.currentTime mapped to ms for perfect synchronization per spec
          const startTimeMs = currentTime * 1000;
          
          const result = faceLandmarkerRef.current.detectForVideo(video, startTimeMs);
          
          if (result.faceLandmarks && result.faceLandmarks.length > 0) {
            lostFramesRef.current = 0;
            const landmarks = result.faceLandmarks[0];
            const blendshapes = result.faceBlendshapes ? result.faceBlendshapes[0].categories : [];

            drawOverlay(landmarks);

            if (cs === "BLINK") {
              const step = challengeStepRef.current;
              
              if (step === "BLINK") {
                // High-Speed Blink Detection
                const blinkLeft = blendshapes.find(b => b.categoryName === 'eyeBlinkLeft')?.score || 0;
                const blinkRight = blendshapes.find(b => b.categoryName === 'eyeBlinkRight')?.score || 0;
                
                if (blinkLeft > 0.5 && blinkRight > 0.5) {
                  challengeStepRef.current = "TURN";
                  updateLiveText("Now slowly TURN your head Left or Right");
                }
              } 
              else if (step === "TURN") {
                // Motion Detection (Head Turn)
                const nose = landmarks[1];
                const leftEyeOuter = landmarks[33];
                const rightEyeOuter = landmarks[263];
                
                const eyeCenter = (leftEyeOuter.x + rightEyeOuter.x) / 2;
                const turnDiff = nose.x - eyeCenter;
                
                // Nose moves significantly left or right
                if (turnDiff < -0.04 || turnDiff > 0.04) {
                  challengeStepRef.current = "DONE";
                  updateState("VERIFYING");
                  updateLiveText("Capturing...");
                  captureAndVerify();
                  return; // Stop loop
                }
              }
            }
          } else if (cs === "BLINK") {
            lostFramesRef.current++;
            if (lostFramesRef.current > 30) {
               updateState("FAILED");
               return;
            }
          } else {
             // Clear overlay if no face
             const canvas = canvasRef.current;
             const ctx = canvas?.getContext("2d");
             if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(detect);
    };

    animFrameRef.current = requestAnimationFrame(detect);
  }

  /* ── Draw Face Overlay ── */
  function drawOverlay(landmarks: Array<{x: number, y: number}>) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 360;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cs = stateRef.current;
    const color = cs === "BLINK" ? "#3B82F6" : "#22C55E";

    ctx.fillStyle = color;
    // Fast render: drawing a subset of landmarks to save compute in React
    const importantLandmarks = [1, 33, 263, 61, 291]; // Nose, Eyes, Mouth
    for (let point of importantLandmarks) {
      const pt = landmarks[point];
      if (!pt) continue;
      ctx.beginPath();
      ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /* ── Final Capture ── */
  async function captureAndVerify() {
    if (!videoRef.current) return;

    // Take a high-quality snapshot
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      // Draw actual video frame
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    }
    
    // Convert to base64 jpg for backend matching
    const base64Image = canvas.toDataURL("image/jpeg", 0.95);

    updateState("SUCCESS");

    setTimeout(() => {
      onCapture(base64Image);
    }, 800);
  }

  /* ── Retry ── */
  function handleRetry() {
    updateState("WAITING");
    runDetectionLoop();
  }

  /* ── UI ── */
  const info = INSTRUCTIONS[state];
  const isActive = state === "BLINK";

  const isBlinkDone = isActive && challengeStepRef.current === "TURN";
  
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
        
        {isActive && (
          <div 
             ref={liveTextRef}
             className="text-lg font-bold mt-2 text-blue-900 animate-pulse"
          >
             Please BLINK your eyes
          </div>
        )}
      </div>

      {/* ── Progress Indicator ── */}
      {(isActive || state === "VERIFYING" || state === "SUCCESS") && (
        <div className="flex justify-center gap-4">
          <div className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
            isBlinkDone || state === "VERIFYING" || state === "SUCCESS"
              ? "bg-green-500 text-white" : "bg-blue-500 text-white animate-pulse"
          }`}>
            1. Blink
          </div>
          <div className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
            state === "VERIFYING" || state === "SUCCESS"
              ? "bg-green-500 text-white" : (isBlinkDone ? "bg-blue-500 text-white animate-pulse" : "bg-gray-200 text-gray-500")
          }`}>
            2. Turn Head
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
