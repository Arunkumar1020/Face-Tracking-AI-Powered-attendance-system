"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import * as faceapi from "face-api.js";
import LivenessFaceCapture from "@/app/components/LivenessFaceCapture";
import MobileBottomNav from "@/app/components/MobileBottomNav";

/* ---------- CONFIG ---------- */
//home:
const CAMPUS_LAT = 10.912951767776038;
const CAMPUS_LNG = 78.0272063956743;
//college GCE:
//const CAMPUS_LAT = 10.694630;
//const CAMPUS_LNG = 78.979179;
//kpr college ;
// const CAMPUS_LAT = 11.0765000;
// const CAMPUS_LNG = 77.1420000;

const ALLOWED_RADIUS = 200; // meters

// ✅ ATTENDANCE RULES
const OFFICE_START_TIME = "09:30";
const LATE_AFTER_TIME = "09:45";
const HALF_DAY_AFTER_TIME = "13:00";

/* ---------- HELPERS ---------- */
function getAttendanceStatus(punchInTime: Date) {
  const timeStr = punchInTime.toTimeString().slice(0, 5);

  if (timeStr <= OFFICE_START_TIME) {
    return { status: "present", remark: "On Time" };
  }

  if (timeStr <= LATE_AFTER_TIME) {
    return { status: "present", remark: "Late Entry" };
  }

  if (timeStr <= HALF_DAY_AFTER_TIME) {
    return { status: "present", remark: "Half Day" };
  }

  return { status: "present", remark: "Late" };
}

function getDistanceInMeters(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
) {
  const R = 6371000;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
    Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLon / 2) ** 2;

  return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

function faceDistance(a: number[], b: number[]) {
  return Math.sqrt(
    a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0)
  );
}

/* ---------- TYPES ---------- */
type PunchAction = "punch_in" | "punch_out";

/* ---------- COMPONENT ---------- */
export default function Attendance() {
  const [employeeId, setEmployeeId] = useState<string | null>(null);
  const [storedFace, setStoredFace] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [showFace, setShowFace] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [punchAction, setPunchAction] = useState<PunchAction | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);

  // Track today's attendance state
  const [hasPunchedIn, setHasPunchedIn] = useState(false);
  const [hasPunchedOut, setHasPunchedOut] = useState(false);
  const [todayRecord, setTodayRecord] = useState<{
    id: string;
    punch_in: string | null;
    punch_out: string | null;
    remark: string | null;
  } | null>(null);

  /* ✅ Load logged-in employee + today's attendance on mount */
  useEffect(() => {
    init();
    loadFaceModels();
  }, []);

  async function loadFaceModels() {
    try {
      const MODEL_URL = "/models";
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
      setModelsLoaded(true);
    } catch (err) {
      console.error("Failed to load generic face models", err);
    }
  }

  async function init() {
    setLoading(true);

    // 1. Get logged-in user
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user?.email) {
      setStatus("❌ Not logged in. Please login first.");
      setLoading(false);
      return;
    }

    // 2. Get employee record by email
    const { data: employee, error } = await supabase
      .from("employees")
      .select("employee_id, face_descriptor")
      .eq("email", user.email)
      .single();

    if (error || !employee) {
      setStatus("❌ Employee record not found for your account.");
      setLoading(false);
      return;
    }

    setEmployeeId(employee.employee_id);
    setStoredFace(employee.face_descriptor);

    if (
      !Array.isArray(employee.face_descriptor) ||
      employee.face_descriptor.length === 0
    ) {
      setStatus(
        "⚠️ Face not registered. Please contact admin before punching attendance."
      );
      setLoading(false);
      return;
    }

    // 3. Check today's attendance record
    const today = new Date().toISOString().split("T")[0];
    const { data: existing } = await supabase
      .from("attendance")
      .select("id, punch_in, punch_out, remark")
      .eq("employee_id", employee.employee_id)
      .eq("date", today)
      .single();

    if (existing) {
      setTodayRecord(existing);
      setHasPunchedIn(!!existing.punch_in);
      setHasPunchedOut(!!existing.punch_out);
    }

    setLoading(false);
  }

  /* ---------- Start Punch (In or Out) ---------- */
  async function startPunch(action: PunchAction) {
    if (!employeeId || !storedFace) return;

    setPunchAction(action);

    if (!navigator.geolocation) {
      setStatus("❌ GPS not supported on this device");
      return;
    }

    setLoading(true);
    setStatus("📍 Verifying campus location...");

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;

        const distance = getDistanceInMeters(
          latitude,
          longitude,
          CAMPUS_LAT,
          CAMPUS_LNG
        );

        if (distance > ALLOWED_RADIUS) {
          setStatus(`❌ Outside allowed area (Dist: ${Math.round(distance)}m)`);
          setLoading(false);
          setPunchAction(null);
          return;
        }

        setStatus("✅ Location verified. Please verify your face");
        setShowFace(true);
        setLoading(false);
      },
      (err) => {
        setStatus(`❌ Location error: ${err.message}`);
        setLoading(false);
        setPunchAction(null);
      },
      {
        enableHighAccuracy: false, // GPS can timeout indoors. Setting to false uses faster Wi-Fi/Cellular signals
        timeout: 30000,            // Give it 30 seconds instead of 15 seconds
        maximumAge: 10000,         // Allow locations cached in the last 10 seconds to speed it up
      }
    );
  }

  /* ---------- Face Verified → Execute Punch ---------- */
  async function handleFaceVerified(imageSrc: string) {
    if (!storedFace || !employeeId || !punchAction) return;

    setLoading(true);
    setStatus("🔍 Verifying face with backend...");

    if (!modelsLoaded) {
      setStatus("❌ Face matching models not loaded yet.");
      setLoading(false);
      setShowFace(false);
      setPunchAction(null);
      return;
    }

    // 1. Create an image element from the base64 output of MediaPipe
    const img = new Image();
    img.src = imageSrc;
    await new Promise((resolve) => {
      img.onload = resolve;
    });

    // 2. Extract descriptor using face-api.js
    const detection = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      setStatus("❌ Could not extract face features for matching. Try again.");
      setLoading(false);
      setShowFace(false);
      setPunchAction(null);
      return;
    }

    const liveDescriptor = Array.from(detection.descriptor);

    // 3. Compare with stored descriptor from the database
    const distance = faceDistance(liveDescriptor, storedFace);
    console.log(`Face match distance: ${distance}`);

    if (distance > 0.6) {
      setStatus("❌ Face mismatch. Attendance denied.");
      setLoading(false);
      setShowFace(false);
      setPunchAction(null);
      return;
    }

    const today = new Date().toISOString().split("T")[0];
    const now = new Date();

    if (punchAction === "punch_in") {
      // ── PUNCH IN ──
      const { data: existing } = await supabase
        .from("attendance")
        .select("id")
        .eq("employee_id", employeeId)
        .eq("date", today)
        .single();

      if (existing) {
        setStatus("⚠️ Already punched in today");
        setLoading(false);
        setShowFace(false);
        setPunchAction(null);
        return;
      }

      const attendanceResult = getAttendanceStatus(now);

      const { error: insertError } = await supabase
        .from("attendance")
        .insert({
          employee_id: employeeId,
          date: today,
          punch_in: now.toISOString(),
          status: attendanceResult.status,
          remark: attendanceResult.remark,
        });

      if (insertError) {
        setStatus(`❌ ${insertError.message}`);
      } else {
        setStatus(`✅ Punched In (${attendanceResult.remark})`);
        setHasPunchedIn(true);
      }
    } else {
      // ── PUNCH OUT ──
      const { data: record } = await supabase
        .from("attendance")
        .select("id, punch_out")
        .eq("employee_id", employeeId)
        .eq("date", today)
        .single();

      if (!record) {
        setStatus("⚠️ No punch-in found for today. Punch in first!");
        setLoading(false);
        setShowFace(false);
        setPunchAction(null);
        return;
      }

      if (record.punch_out) {
        setStatus("⚠️ Already punched out today");
        setLoading(false);
        setShowFace(false);
        setPunchAction(null);
        return;
      }

      const { error: updateError } = await supabase
        .from("attendance")
        .update({ punch_out: now.toISOString() })
        .eq("id", record.id);

      if (updateError) {
        setStatus(`❌ ${updateError.message}`);
      } else {
        setStatus("✅ Punched Out successfully");
        setHasPunchedOut(true);
      }
    }

    setShowFace(false);
    setLoading(false);
    setPunchAction(null);

    // Refresh today's record
    await refreshTodayRecord();
  }

  async function refreshTodayRecord() {
    if (!employeeId) return;
    const today = new Date().toISOString().split("T")[0];
    const { data } = await supabase
      .from("attendance")
      .select("id, punch_in, punch_out, remark")
      .eq("employee_id", employeeId)
      .eq("date", today)
      .single();

    if (data) {
      setTodayRecord(data);
      setHasPunchedIn(!!data.punch_in);
      setHasPunchedOut(!!data.punch_out);
    }
  }

  /* ---------- Format time for display ---------- */
  function formatTime(isoString: string | null) {
    if (!isoString) return "--:--";
    return new Date(isoString).toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
  }

  /* ---------- UI ---------- */
  const isPunchInDisabled = hasPunchedIn || loading;
  const isPunchOutDisabled = !hasPunchedIn || hasPunchedOut || loading;

  return (
    <div className="min-h-screen bg-blue-50 flex items-center justify-center px-4 pb-20">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-xl p-6 space-y-5">
        <h2 className="text-2xl font-bold text-center text-blue-700">
          Attendance
        </h2>

        {/* ── Status Message ── */}
        {status && (
          <div className="text-center text-sm bg-blue-100 text-blue-900 p-3 rounded-lg">
            {status}
          </div>
        )}

        {/* ── Today's Summary ── */}
        {todayRecord && (
          <div className="bg-gray-50 rounded-xl p-4 space-y-2">
            <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
              Today&apos;s Record
            </h3>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Punch In</span>
              <span className="font-medium text-green-700">
                {formatTime(todayRecord.punch_in)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Punch Out</span>
              <span className="font-medium text-red-600">
                {formatTime(todayRecord.punch_out)}
              </span>
            </div>
            {todayRecord.remark && (
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Remark</span>
                <span className="font-medium text-blue-700">
                  {todayRecord.remark}
                </span>
              </div>
            )}
          </div>
        )}

        {/* ── Punch In / Punch Out Buttons ── */}
        {!showFace && !loading && (
          <div className="flex gap-3">
            <button
              onClick={() => startPunch("punch_in")}
              disabled={isPunchInDisabled}
              className={`flex-1 py-3 rounded-lg text-lg font-semibold transition-all ${isPunchInDisabled
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-green-600 text-white hover:bg-green-700 active:scale-95"
                }`}
            >
              Punch In
            </button>

            <button
              onClick={() => startPunch("punch_out")}
              disabled={isPunchOutDisabled}
              className={`flex-1 py-3 rounded-lg text-lg font-semibold transition-all ${isPunchOutDisabled
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-red-600 text-white hover:bg-red-700 active:scale-95"
                }`}
            >
              Punch Out
            </button>
          </div>
        )}

        {/* ── Loading Spinner ── */}
        {loading && !showFace && (
          <div className="text-center py-4">
            <div className="inline-block w-8 h-8 border-4 border-blue-300 border-t-blue-600 rounded-full animate-spin" />
          </div>
        )}

        {/* ── Face Verification with Liveness ── */}
        {showFace && (
          <LivenessFaceCapture
            onCapture={handleFaceVerified}
            onCancel={() => {
              setShowFace(false);
              setPunchAction(null);
              setStatus(null);
            }}
            timeoutSeconds={10}
          />
        )}
      </div>

      <MobileBottomNav />
    </div>
  );
}
