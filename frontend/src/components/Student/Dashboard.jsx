// frontend/src/Components/Student/Dashboard.jsx
import React, { useContext, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Box, Card, CardContent, Typography, CircularProgress, Grid, Chip, Stack, Divider,
  Button, Tooltip, Table, TableHead, TableRow, TableCell, TableBody, Fab, Accordion,
  AccordionSummary, AccordionDetails, Checkbox, FormControlLabel,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
// ... (keep all your other icon imports)
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import ArrowUpwardRoundedIcon from "@mui/icons-material/ArrowUpwardRounded";
import ArrowDownwardRoundedIcon from "@mui/icons-material/ArrowDownwardRounded";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import ChatRoundedIcon from "@mui/icons-material/ChatRounded";

import ChatWidget from "./Chat/ChatWidget";
import Navbar from "../Navbar";
import { AuthContext } from "../../context/AuthContext";

// --- Sub-Components (Risk UI, unchanged) ---
const riskColor = (riskCategory) => { /* ... same as before ... */ if (!riskCategory) return { color: "default", hex: "#9ca3af" }; const txt = riskCategory.toLowerCase(); if (txt.includes("high")) return { color: "error", hex: "#ef4444" }; if (txt.includes("medium")) return { color: "warning", hex: "#f59e0b" }; return { color: "success", hex: "#10b981" };};
const CircularGauge = ({ value = 0, colorHex = "#2563eb", size = 140 }) => { /* ... same as before ... */ const safeVal = Math.max(0, Math.min(100, Number(value) || 0)); return (<Box sx={{ position: "relative", display: "inline-flex" }}> <CircularProgress variant="determinate" value={100} size={size} thickness={5} sx={{ color: "#e5e7eb" }}/> <CircularProgress variant="determinate" value={safeVal} size={size} thickness={5} sx={{ color: colorHex, position: "absolute", left: 0 }}/> <Box sx={{ top: 0, left: 0, bottom: 0, right: 0, position: "absolute", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 0.5, }} > <Typography variant="h4" sx={{ fontWeight: 700, lineHeight: 1 }}> {safeVal.toFixed(0)}% </Typography> <Typography variant="caption" color="text.secondary"> Risk </Typography> </Box> </Box>);};
const StatCard = ({ title, value, subtitle }) => ( /* ... same as before ... */ <Card><CardContent><Typography variant="subtitle2" color="text.secondary">{title}</Typography><Typography variant="h4" sx={{ fontWeight: 700 }}>{value}</Typography>{subtitle && (<Typography variant="caption" color="text.secondary">{subtitle}</Typography>)}</CardContent></Card>);

// +++ NEW: Personalized Study Plan Component +++
const StudyPlan = ({ plan, loading, error }) => {
  if (loading) {
    return (
      <Card>
        <CardContent sx={{ textAlign: 'center' }}>
          <CircularProgress size={30} />
          <Typography sx={{ mt: 1 }} color="text.secondary">Generating your study plan...</Typography>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ borderLeft: '4px solid #f59e0b' }}>
        <CardContent><Typography color="text.secondary">{error}</Typography></CardContent>
      </Card>
    );
  }

  if (!plan || plan.length === 0) {
    return (
      <Card sx={{ borderLeft: '4px solid #10b981' }}>
        <CardContent>
          <Typography sx={{ fontWeight: 'bold' }}>No Study Plan Needed!</Typography>
          <Typography color="text.secondary">All your recent subject scores are above 85%. Keep up the great work! 🎉</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
          Your Personalized Study Plan
        </Typography>
        {plan.map((subjectPlan) => (
          <Accordion key={subjectPlan.subject} defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Stack direction="row" spacing={2} alignItems="center">
                <Typography sx={{ fontWeight: 600 }}>{subjectPlan.subject}</Typography>
                <Chip label={`Your Score: ${subjectPlan.percentage.toFixed(1)}%`} color="warning" size="small"/>
              </Stack>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0 }}>
              {subjectPlan.chapters.length > 0 ? (
                subjectPlan.chapters.map((chapter, idx) => (
                  <Box key={idx} sx={{ mb: 1.5, px: 2 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>{chapter.title}</Typography>
                    {chapter.topics.map((topic, topicIdx) => (
                      <FormControlLabel
                        key={topicIdx}
                        control={<Checkbox size="small" />}
                        label={topic}
                        sx={{ display: 'block', ml: 1 }}
                      />
                    ))}
                  </Box>
                ))
              ) : (
                <Typography sx={{ px: 2, pb: 2 }} color="text.secondary">
                  Syllabus topics for this subject could not be loaded. Please ensure the syllabus has been uploaded by your teacher.
                </Typography>
              )}
            </AccordionDetails>
          </Accordion>
        ))}
      </CardContent>
    </Card>
  );
};


const Dashboard = () => {
  const { user } = useContext(AuthContext);
  // States for risk analysis (unchanged)
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState({ avgRisk: 0, highRiskCount: 0, mediumRiskCount: 0, lowRiskCount: 0 });
  const [isChatOpen, setIsChatOpen] = useState(false);

  // +++ NEW: States for the study plan +++
  const [studyPlan, setStudyPlan] = useState([]);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState("");

  const loadRiskData = async () => { /* ... same as your 'load' function ... */ try { setLoading(true); setErr(""); const apiUrl = import.meta.env.VITE_API_URL; if (!apiUrl) { setErr("API URL missing."); return; } const res = await axios.get(`${apiUrl}/api/student-risks/me`, { params: { email: user?.email } }); setLatest(res.data?.latest || null); setHistory(res.data?.history || []); setSummary({ avgRisk: Number(res.data?.avgRisk || 0), highRiskCount: Number(res.data?.highRiskCount || 0), mediumRiskCount: Number(res.data?.mediumRiskCount || 0), lowRiskCount: Number(res.data?.lowRiskCount || 0) }); } catch (e) { console.error("Dashboard load error:", e); setErr(e.response?.data?.message || "Failed to load analytics."); } finally { setLoading(false); } };

  // +++ NEW: Function to generate the study plan +++
  const generateStudyPlan = async () => {
    if (!user?.email) return;

    setPlanLoading(true);
    setPlanError("");
    setStudyPlan([]);

    try {
      const marksApiUrl = import.meta.env.VITE_API_URL;
      const nlpApiUrl = import.meta.env.VITE_ML_SERVER_URL; // Assuming your python server URL is here

      // 1. Fetch latest marks from Express backend
      const marksRes = await axios.get(`${marksApiUrl}/api/marks/me`, {
        params: { email: user.email },
      });
      const marksData = marksRes.data;

      // 2. Identify subjects with scores below 85%
      const lowScoringSubjects = marksData.filter(
        (mark) => mark.percentage < 85
      );

      if (lowScoringSubjects.length === 0) {
        setPlanLoading(false);
        return; // Exit early if all scores are good
      }

      // 3. For each low-scoring subject, fetch its topics from Python backend
      const planPromises = lowScoringSubjects.map(async (mark) => {
        try {
          const topicsRes = await axios.get(`${nlpApiUrl}/syllabus/topics/${mark.subject}`);
          return {
            ...mark, // Contains subject, percentage, etc.
            chapters: topicsRes.data.chapters,
          };
        } catch (topicError) {
          console.error(`Could not fetch topics for ${mark.subject}:`, topicError);
          return { ...mark, chapters: [] }; // Return subject info even if topics fail
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
  };

  useEffect(() => {
    if (user?.email) {
      loadAllData();
    }
  }, [user?.email]);

  const trend = useMemo(() => { /* ... same as before ... */ if (!history || history.length < 2) return null; const [latestEntry, prevEntry] = history; const curr = Number(latestEntry?.predicted_risk_percentage || 0); const prev = Number(prevEntry?.predicted_risk_percentage || 0); const delta = curr - prev; return { delta, direction: delta === 0 ? "flat" : delta > 0 ? "up" : "down", }; }, [history]);
  const latestColor = riskColor(latest?.risk_category);
  const lastUpdated = latest?.createdAt ? new Date(latest.createdAt).toLocaleString() : null;

  return (
    <>
      <Navbar />

      <Box sx={{ px: { xs: 2, md: 3 }, py: 3, bgcolor: "#f8fafc", minHeight: "calc(100vh - 64px)" }}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>Welcome{user?.name ? `, ${user.name}` : ""} 👋</Typography>
          <Typography variant="body2" color="text.secondary">Here's your latest academic analysis and personalized study plan.</Typography>
        </Box>

        <Stack direction="row" alignItems="center" justifyContent="flex-end" sx={{ mb: 2 }}>
          <Button variant="outlined" onClick={loadAllData} disabled={loading || planLoading} startIcon={<RefreshRoundedIcon />}>
            {(loading || planLoading) ? "Refreshing..." : "Refresh All"}
          </Button>
        </Stack>

        {err && ( <Card sx={{ mb: 2, borderLeft: "4px solid #ef4444" }}><CardContent><Typography color="error">{err}</Typography></CardContent></Card> )}
        
        {/* --- NEW: Study Plan Section --- */}
        <Box sx={{ mb: 3 }}>
          <StudyPlan plan={studyPlan} loading={planLoading} error={planError} />
        </Box>

        {/* --- Risk Analysis Section (unchanged UI) --- */}
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>
              Academic Risk Analysis
            </Typography>
            <Divider sx={{ my: 1 }} />
          </Grid>
          
          <Grid item xs={12} md={5}>
            {/* ... Your original Risk Snapshot card UI ... */}
            <Card sx={{ height: "100%" }}> <CardContent> <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}> Latest Risk Snapshot </Typography> <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="center"> <CircularGauge value={Number(latest?.predicted_risk_percentage || 0)} colorHex={latestColor.hex} /> <Box sx={{ flex: 1 }}> <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}> <Chip label={latest?.risk_category || "No Data"} color={latestColor.color} size="small" /> {trend && trend.direction !== "flat" && ( <Chip size="small" icon={ trend.direction === "up" ? ( <ArrowUpwardRoundedIcon /> ) : ( <ArrowDownwardRoundedIcon /> ) } label={`${ trend.delta > 0 ? "+" : "" }${trend.delta.toFixed(1)}% vs last`} color={trend.direction === "up" ? "error" : "success"} variant="outlined" /> )} </Stack> <Typography variant="body2" color="text.secondary"> {lastUpdated ? `Updated: ${lastUpdated}` : "No recent analysis."} </Typography> </Box> </Stack> </CardContent> </Card>
          </Grid>
          
          <Grid item xs={12} md={7}>
            {/* ... Your original Stat Cards UI ... */}
             <Grid container spacing={2} sx={{ height: "100%" }}> <Grid item xs={12} sm={6} md={4}> <StatCard title="Average Risk" value={ summary.avgRisk ? `${Number(summary.avgRisk || 0).toFixed(2)}%` : "—" } /> </Grid> <Grid item xs={12} sm={6} md={4}> <StatCard title="High Risk Events" value={summary.highRiskCount || 0} /> </Grid> <Grid item xs={12} sm={6} md={4}> <StatCard title="Medium Risk Events" value={summary.mediumRiskCount || 0} /> </Grid> </Grid>
          </Grid>

          <Grid item xs={12}>
            <Card sx={{ mt: 2 }}>
              {/* ... Your original History Table UI ... */}
              <CardContent> <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}> Recent History </Typography> {!loading && history.length === 0 ? (<Typography variant="body2">No history to display.</Typography>) : (<Box sx={{ width: "100%", overflowX: "auto" }}> <Table size="small"> <TableHead> <TableRow> <TableCell>Date</TableCell> <TableCell align="right">Risk %</TableCell> <TableCell>Category</TableCell> </TableRow> </TableHead> <TableBody> {history.slice(0, 5).map((h) => { const catColor = riskColor(h.risk_category); return ( <TableRow key={h._id || h.id}> <TableCell>{new Date(h.createdAt).toLocaleString()}</TableCell> <TableCell align="right">{Number(h.predicted_risk_percentage || 0).toFixed(2)}%</TableCell> <TableCell><Chip label={h.risk_category} color={catColor.color} size="small" /></TableCell> </TableRow> ); })} </TableBody> </Table> </Box>)} </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      <Fab color="primary" aria-label="open chat" onClick={() => setIsChatOpen(true)} sx={{ position: "fixed", bottom: { xs: 24, md: 32 }, right: { xs: 24, md: 32 } }}>
        <ChatRoundedIcon />
      </Fab>

      <ChatWidget open={isChatOpen} onClose={() => setIsChatOpen(false)} studentContext={latest}/>
    </>
  );
};

export default Dashboard;