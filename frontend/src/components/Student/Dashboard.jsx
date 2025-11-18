// frontend/src/Components/Student/Dashboard.jsx
import React, { useContext, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { FiRefreshCw, FiClock, FiArrowUpRight, FiArrowDownRight, FiInfo, FiMessageSquare, FiUserPlus, FiCheckCircle } from "react-icons/fi";

import ChatWidget from "./Chat/ChatWidget";
import Navbar from "../Navbar";
import { AuthContext } from "../../context/AuthContext";

// --- Risk color helper (kept same logic, Tailwind classes) ---
const riskColor = (riskCategory) => {
    if (!riskCategory)
        return {
            badge: "bg-slate-200 text-slate-700",
            hex: "#9ca3af",
        };

    const txt = riskCategory.toLowerCase();
    if (txt.includes("high"))
        return {
            badge: "bg-red-100 text-red-700",
            hex: "#ef4444",
        };
    if (txt.includes("medium"))
        return {
            badge: "bg-amber-100 text-amber-700",
            hex: "#f59e0b",
        };
    return {
        badge: "bg-emerald-100 text-emerald-700",
        hex: "#10b981",
    };
};

// --- Circular Gauge (replaces MUI CircularProgress) ---
const CircularGauge = ({ value = 0, colorHex = "#2563eb", size = 140 }) => {
    const safeVal = Math.max(0, Math.min(100, Number(value) || 0));
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (safeVal / 100) * circumference;

    return (
        <div
            className="relative inline-flex items-center justify-center"
            style={{ width: size, height: size }}
        >
            <svg
                className="transform -rotate-90"
                width={size}
                height={size}
                viewBox="0 0 100 100"
            >
                {/* Background circle */}
                <circle
                    cx="50"
                    cy="50"
                    r={radius}
                    stroke="#e5e7eb"
                    strokeWidth="6"
                    fill="none"
                />
                {/* Progress circle */}
                <circle
                    cx="50"
                    cy="50"
                    r={radius}
                    stroke={colorHex}
                    strokeWidth="6"
                    fill="none"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <p className="text-3xl font-semibold leading-none">{safeVal.toFixed(0)}%</p>
                <p className="text-xs text-slate-500 mt-1">Risk</p>
            </div>
        </div>
    );
};

// --- Stat Card ---
const StatCard = ({ title, value, subtitle }) => (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4 flex flex-col gap-1">
        <p className="text-xs font-medium text-slate-500">{title}</p>
        <p className="text-2xl font-semibold text-slate-900">{value}</p>
        {subtitle && <p className="text-[11px] text-slate-500 mt-1">{subtitle}</p>}
    </div>
);

// --- Personalized Study Plan Component ---
const StudyPlan = ({ plan, loading, error }) => {
    if (loading) {
        return (
            <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6 text-center">
                <div className="w-6 h-6 border-2 border-slate-300 border-t-slate-500 rounded-full animate-spin mx-auto" />
                <p className="text-sm text-slate-500 mt-2">Generating your study plan...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-white rounded-xl shadow-sm border-l-4 border-amber-500 p-4">
                <p className="text-sm text-slate-600">{error}</p>
            </div>
        );
    }

    if (!plan || plan.length === 0) {
        return (
            <div className="bg-white rounded-xl shadow-sm border-l-4 border-emerald-500 p-4">
                <p className="text-sm font-semibold text-slate-900">No Study Plan Needed!</p>
                <p className="text-xs text-slate-500 mt-1">All your recent subject scores are above 85%. Keep up the great work! 🎉</p>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4 md:p-5">
            <div className="flex items-center justify-between mb-3">
                <p className="text-base font-semibold text-slate-900">Your Personalized Study Plan</p>
            </div>

            <div className="space-y-3">
                {plan.map((subjectPlan) => (
                    <details
                        key={subjectPlan.subject}
                        open
                        className="border border-slate-200 rounded-lg overflow-hidden"
                    >
                        <summary className="flex items-center justify-between px-3 py-2.5 cursor-pointer bg-slate-50 hover:bg-slate-100">
                            <div className="flex items-center gap-3">
                                <p className="text-sm font-semibold text-slate-900">{subjectPlan.subject}</p>
                                <span className="inline-flex items-center rounded-full bg-amber-100 text-amber-700 text-[11px] px-2 py-0.5 font-medium">
                                    Your Score: {subjectPlan.percentage.toFixed(1)}%
                                </span>
                            </div>
                            <span className="text-slate-400 text-xs">Expand</span>
                        </summary>

                        <div className="px-3 py-2 space-y-2">
                            {subjectPlan.chapters.length > 0 ? (
                                subjectPlan.chapters.map((chapter, idx) => (
                                    <div
                                        key={idx}
                                        className="mb-1.5"
                                    >
                                        <p className="text-xs font-semibold text-slate-800 mb-0.5">{chapter.title}</p>
                                        <div className="space-y-1">
                                            {chapter.topics.map((topic, topicIdx) => (
                                                <label
                                                    key={topicIdx}
                                                    className="flex items-center gap-2 text-xs text-slate-700"
                                                >
                                                    <input
                                                        type="checkbox"
                                                        className="w-3.5 h-3.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                                                    />
                                                    <span>{topic}</span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <p className="text-xs text-slate-500 pb-1">
                                    Syllabus topics for this subject could not be loaded. Please ensure the syllabus has been uploaded by your
                                    teacher.
                                </p>
                            )}
                        </div>
                    </details>
                ))}
            </div>
        </div>
    );
};

const Dashboard = () => {
    const { user } = useContext(AuthContext);

    // Risk analysis state
    const [loading, setLoading] = useState(false);
    const [err, setErr] = useState("");
    const [latest, setLatest] = useState(null);
    const [history, setHistory] = useState([]);
    const [summary, setSummary] = useState({
        avgRisk: 0,
        highRiskCount: 0,
        mediumRiskCount: 0,
        lowRiskCount: 0,
    });
    const [isChatOpen, setIsChatOpen] = useState(false);

    // Study plan state
    const [studyPlan, setStudyPlan] = useState([]);
    const [planLoading, setPlanLoading] = useState(false);
    const [planError, setPlanError] = useState("");

    // Mentor request state
    const [mentorInfo, setMentorInfo] = useState(null);
    const [mentorStatus, setMentorStatus] = useState("none");
    // const [mentorStatus, setMentorStatus] = useState(null); // "pending" or null
    const [requestLoading, setRequestLoading] = useState(false);

    const loadMentorStatus = async () => {
        try {
            const apiUrl = import.meta.env.VITE_API_URL;
            const res = await axios.get(`${apiUrl}/api/student/mentor-status/${user.id}`);

            setMentorStatus(res.data.status || "none");
            setMentorInfo(res.data.mentor || null);
        } catch (e) {
            console.error("Failed to load mentor status", e);
        }
    };

    const loadRiskData = async () => {
        try {
            setLoading(true);
            setErr("");
            const apiUrl = import.meta.env.VITE_API_URL;
            if (!apiUrl) {
                setErr("API URL missing.");
                return;
            }
            const res = await axios.get(`${apiUrl}/api/student-risks/me`, {
                params: { email: user?.email },
            });
            setLatest(res.data?.latest || null);
            setHistory(res.data?.history || []);
            setSummary({
                avgRisk: Number(res.data?.avgRisk || 0),
                highRiskCount: Number(res.data?.highRiskCount || 0),
                mediumRiskCount: Number(res.data?.mediumRiskCount || 0),
                lowRiskCount: Number(res.data?.lowRiskCount || 0),
            });
        } catch (e) {
            console.error("Dashboard load error:", e);
            setErr(e.response?.data?.message || "Failed to load analytics.");
        } finally {
            setLoading(false);
        }
    };

    const generateStudyPlan = async () => {
        if (!user?.email) return;

        setPlanLoading(true);
        setPlanError("");
        setStudyPlan([]);

        try {
            const marksApiUrl = import.meta.env.VITE_API_URL;
            const nlpApiUrl = import.meta.env.VITE_ML_SERVER_URL;

            const marksRes = await axios.get(`${marksApiUrl}/api/marks/me`, {
                params: { email: user.email },
            });
            const marksData = marksRes.data;

            const lowScoringSubjects = marksData.filter((mark) => mark.percentage < 85);

            if (lowScoringSubjects.length === 0) {
                setPlanLoading(false);
                return;
            }

            const planPromises = lowScoringSubjects.map(async (mark) => {
                try {
                    const topicsRes = await axios.get(`${nlpApiUrl}/syllabus/topics/${mark.subject}`);
                    return {
                        ...mark,
                        chapters: topicsRes.data.chapters,
                    };
                } catch (topicError) {
                    console.error(`Could not fetch topics for ${mark.subject}:`, topicError);
                    return { ...mark, chapters: [] };
                }
            });

            const fullPlan = await Promise.all(planPromises);
            setStudyPlan(fullPlan);
        } catch (e) {
            console.error("Study plan generation error:", e);
            setPlanError("Could not generate your study plan. Please try again later.");
        } finally {
            setPlanLoading(false);
        }
    };

    const loadAllData = () => {
        loadRiskData();
        generateStudyPlan();
        loadMentorStatus();
    };

    const requestMentor = async () => {
        try {
            setRequestLoading(true);
            const apiUrl = import.meta.env.VITE_API_URL;

            await axios.post(`${apiUrl}/api/student/mentor-request`, {
                userId: user.id, // FIXED HERE
            });

            setMentorStatus("pending");
        } catch (e) {
            console.error("Mentor request error:", e);
        } finally {
            setRequestLoading(false);
        }
    };

    useEffect(() => {
        if (user?.email) {
            loadAllData();
        }
    }, [user?.email]);

    const trend = useMemo(() => {
        if (!history || history.length < 2) return null;
        const [latestEntry, prevEntry] = history;
        const curr = Number(latestEntry?.predicted_risk_percentage || 0);
        const prev = Number(prevEntry?.predicted_risk_percentage || 0);
        const delta = curr - prev;
        return {
            delta,
            direction: delta === 0 ? "flat" : delta > 0 ? "up" : "down",
        };
    }, [history]);

    const latestColor = riskColor(latest?.risk_category);
    const lastUpdated = latest?.createdAt ? new Date(latest.createdAt).toLocaleString() : null;

    return (
        <>
            <Navbar />

            <div className="px-3 md:px-6 py-4 bg-slate-50 min-h-[calc(100vh-64px)]">
                {/* Header */}
                <div className="mb-3">
                    <h1 className="text-xl md:text-2xl font-semibold text-slate-900">Welcome{user?.name ? `, ${user.name}` : ""} 👋</h1>
                    <p className="text-xs md:text-sm text-slate-500">Here's your latest academic analysis and personalized study plan.</p>
                </div>

                {/* Refresh + Ask Mentor */}
                <div className="flex items-center justify-end gap-2 mb-3">
                    <button
                        onClick={loadAllData}
                        disabled={loading || planLoading}
                        className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-1.5 text-xs md:text-sm font-medium text-slate-700 bg-white hover:bg-slate-50 disabled:opacity-60 disabled:cursor-not-allowed"
                    >
                        <FiRefreshCw className="w-4 h-4" />
                        {loading || planLoading ? "Refreshing..." : "Refresh All"}
                    </button>

                    {/* Mentor Button */}
                    {mentorStatus === "approved" && mentorInfo ? (
                        <div className="inline-flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-3 py-1.5 rounded-lg text-xs md:text-sm font-medium">
                            <FiCheckCircle className="w-4 h-4" />
                            Mentor: {mentorInfo.name}
                        </div>
                    ) : mentorStatus === "pending" ? (
                        <div className="inline-flex items-center gap-2 bg-amber-500 text-white px-3 py-1.5 rounded-lg text-xs md:text-sm font-medium">
                            <FiClock className="w-4 h-4" />
                            Mentor Requested
                        </div>
                    ) : (
                        <button
                            onClick={requestMentor}
                            disabled={requestLoading}
                            className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1.5 rounded-lg text-xs md:text-sm font-medium disabled:opacity-60"
                        >
                            <FiUserPlus className="w-4 h-4" />
                            Ask for Mentor
                        </button>
                    )}
                </div>

                {/* Error card */}
                {err && (
                    <div className="bg-white rounded-xl shadow-sm border-l-4 border-red-500 p-3 mb-3">
                        <p className="text-sm text-red-600">{err}</p>
                    </div>
                )}

                {/* Study Plan */}
                <div className="mb-4">
                    <StudyPlan
                        plan={studyPlan}
                        loading={planLoading}
                        error={planError}
                    />
                </div>

                {/* Academic Risk Analysis */}
                <div className="mb-2">
                    <p className="text-sm font-semibold text-slate-900">Academic Risk Analysis</p>
                    <div className="h-px bg-slate-200 mt-1" />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-12 gap-3 md:gap-4">
                    {/* Latest Risk Snapshot */}
                    <div className="md:col-span-5">
                        <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4 h-full">
                            <p className="text-sm font-semibold text-slate-900 mb-2 flex items-center gap-1.5">
                                Latest Risk Snapshot
                                <FiInfo className="w-3 h-3 text-slate-400" />
                            </p>

                            {!latest ? (
                                <p className="text-xs text-slate-500">No recent analysis.</p>
                            ) : (
                                <div className="flex flex-col sm:flex-row items-center gap-3">
                                    <CircularGauge
                                        value={Number(latest?.predicted_risk_percentage || 0)}
                                        colorHex={latestColor.hex}
                                    />
                                    <div className="flex-1 space-y-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span
                                                className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${latestColor.badge}`}
                                            >
                                                {latest?.risk_category || "No Data"}
                                            </span>
                                            {trend && trend.direction !== "flat" && (
                                                <span
                                                    className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] ${
                                                        trend.direction === "up"
                                                            ? "border-red-300 text-red-600"
                                                            : "border-emerald-300 text-emerald-600"
                                                    }`}
                                                >
                                                    {trend.direction === "up" ? (
                                                        <FiArrowUpRight className="w-3 h-3" />
                                                    ) : (
                                                        <FiArrowDownRight className="w-3 h-3" />
                                                    )}
                                                    {`${trend.delta > 0 ? "+" : ""}${trend.delta.toFixed(1)}% vs last`}
                                                </span>
                                            )}
                                        </div>

                                        <p className="text-xs text-slate-500">{lastUpdated ? `Updated: ${lastUpdated}` : "No recent analysis."}</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Stat cards */}
                    <div className="md:col-span-7">
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 h-full">
                            <StatCard
                                title="Average Risk"
                                value={summary.avgRisk ? `${Number(summary.avgRisk || 0).toFixed(2)}%` : "—"}
                            />
                            <StatCard
                                title="High Risk Events"
                                value={summary.highRiskCount || 0}
                            />
                            <StatCard
                                title="Medium Risk Events"
                                value={summary.mediumRiskCount || 0}
                            />
                        </div>
                    </div>

                    {/* History table */}
                    <div className="md:col-span-12">
                        <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4 mt-2">
                            <p className="text-sm font-semibold text-slate-900 mb-2">Recent History</p>
                            {!loading && history.length === 0 ? (
                                <p className="text-xs text-slate-500">No history to display.</p>
                            ) : (
                                <div className="w-full overflow-x-auto">
                                    <table className="min-w-full text-left text-xs">
                                        <thead>
                                            <tr className="border-b border-slate-200 bg-slate-50">
                                                <th className="px-3 py-2 font-medium text-slate-600">Date</th>
                                                <th className="px-3 py-2 font-medium text-slate-600 text-right">Risk %</th>
                                                <th className="px-3 py-2 font-medium text-slate-600">Category</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {history.slice(0, 5).map((h) => {
                                                const c = riskColor(h.risk_category);
                                                return (
                                                    <tr
                                                        key={h._id || h.id}
                                                        className="border-b border-slate-100 last:border-0"
                                                    >
                                                        <td className="px-3 py-2 text-[11px] text-slate-700 whitespace-nowrap">
                                                            {new Date(h.createdAt).toLocaleString()}
                                                        </td>
                                                        <td className="px-3 py-2 text-[11px] text-right text-slate-800">
                                                            {Number(h.predicted_risk_percentage || 0).toFixed(2)}%
                                                        </td>
                                                        <td className="px-3 py-2">
                                                            <span
                                                                className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${c.badge}`}
                                                            >
                                                                {h.risk_category}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Chat FAB */}
            <button
                onClick={() => setIsChatOpen(true)}
                className="fixed bottom-6 right-6 md:bottom-8 md:right-8 w-12 h-12 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white flex items-center justify-center shadow-lg"
                aria-label="open chat"
            >
                <FiMessageSquare className="w-5 h-5" />
            </button>

            <ChatWidget
                open={isChatOpen}
                onClose={() => setIsChatOpen(false)}
                studentContext={latest}
            />
        </>
    );
};

export default Dashboard;
