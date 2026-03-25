"use client";

import * as faceapi from "face-api.js";
import { useEffect, useRef, useState } from "react";
import { supabase } from "@/lib/supabase";

type Employee = {
  id: string;
  employee_id: string;
  name: string;
  department: string;
};

type FaceRegisterProps = {
  employee: Employee;
  onSuccess: (descriptor: number[]) => Promise<void>;
  onClose: () => void;
};

export default function FaceRegister({
  employee,
  onSuccess,
  onClose,
}: FaceRegisterProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [loading, setLoading] = useState(false);
  const [modelsReady, setModelsReady] = useState(false);
  const [instruction, setInstruction] = useState(
    "Position your face clearly inside the camera frame."
  );
  const [error, setError] = useState<string | null>(null);

  /* ---------- LOAD MODELS + CAMERA ---------- */
  useEffect(() => {
    let stream: MediaStream;

    async function init() {
      try {
        setInstruction("⏳ Loading face detection models...");

        const MODEL_URL = "/models";
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);

        setModelsReady(true);
        setInstruction("📷 Allow camera access and look straight.");

        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error(err);
        setError("Camera or model loading failed.");
        setInstruction("❌ Unable to access camera.");
      }
    }

    init();

    return () => {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  /* ---------- REGISTER FACE ---------- */
  async function registerFace() {
    if (!videoRef.current || loading || !modelsReady) return;

    setLoading(true);
    setError(null);
    setInstruction("📸 Capturing face… Please stay still.");

    const detection = await faceapi
      .detectSingleFace(
        videoRef.current,
        new faceapi.TinyFaceDetectorOptions()
      )
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      setInstruction("❌ No face detected. Try again.");
      setLoading(false);
      return;
    }

    const descriptor = Array.from(detection.descriptor);

    setInstruction("💾 Saving face data securely…");

    const { data: updated, error } = await supabase
      .from("employees")
      .update({ face_descriptor: descriptor })
      .eq("employee_id", employee.employee_id)
      .select("employee_id")
      .single();

    setLoading(false);

    if (error) {
      console.error("Supabase update error:", error);
      setError(`Failed to save: ${error.message}`);
      setInstruction("❌ Face registration failed.");
      return;
    }

    if (!updated) {
      console.error("No rows updated for employee_id:", employee.employee_id);
      setError("No matching employee found to update.");
      setInstruction("❌ Face registration failed.");
      return;
    }

    setInstruction("✅ Face registered successfully!");
    await onSuccess(descriptor);
    setTimeout(onClose, 900);
  }

  /* ---------- UI ---------- */
  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center px-4">
      <div className="bg-white w-full max-w-md rounded-xl shadow-2xl p-6 space-y-4">
        {/* Header */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Face Registration
          </h2>
          <p className="text-sm text-gray-500">
            Employee: <span className="font-medium">{employee.name}</span>
          </p>
        </div>

        {/* Instruction */}
        <div className="rounded-lg bg-blue-50 border border-blue-200 p-3 text-sm text-blue-800">
          {instruction}
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Camera */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full rounded-lg border border-gray-300"
        />

        {/* Actions */}
        <div className="flex gap-3 pt-2">
          <button
            onClick={registerFace}
            disabled={loading || !modelsReady}
            className={`flex-1 py-2 rounded-lg font-medium transition ${
              loading || !modelsReady
                ? "bg-gray-400 cursor-not-allowed text-white"
                : "bg-blue-600 hover:bg-blue-700 text-white"
            }`}
          >
            {loading ? "Processing…" : "Capture Face"}
          </button>

          <button
            onClick={onClose}
            disabled={loading}
            className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-100"
          >
            Cancel
          </button>
        </div>

        {/* Loading Indicator */}
        {loading && (
          <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
            Saving face data…
          </div>
        )}
      </div>
    </div>
  );
}
