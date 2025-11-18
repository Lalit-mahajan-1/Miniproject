import React, { useState, useMemo, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  Button,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Stack,
  Box,
  Divider,
  LinearProgress,
  Chip,
  Tooltip,
  TextField,
  Grid, // NEW: Import Grid for layout
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SaveIcon from "@mui/icons-material/Save";
import SchoolIcon from "@mui/icons-material/School";
import SummarizeIcon from "@mui/icons-material/Summarize";
import DownloadIcon from "@mui/icons-material/Download";
import InsightsIcon from "@mui/icons-material/Insights";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import ReportRoundedIcon from "@mui/icons-material/ReportRounded";
import Papa from "papaparse";
import ResultsTable from "./ResultsTable";
import "./MarksUpload.css";

// CountUp Helper (no changes needed)
const CountUp = ({ value = 0, decimals = 0, duration = 1200, suffix = "" }) => {
    // ... same code as before ...
    const [display, setDisplay] = useState(0); const rafRef = useRef(null); useEffect(() => { let start; const from = 0; const to = Number(value) || 0; const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3); const step = (ts) => { if (!start) start = ts; const progress = Math.min((ts - start) / duration, 1); const eased = easeOutCubic(progress); const current = from + (to - from) * eased; setDisplay(current); if (progress < 1) rafRef.current = requestAnimationFrame(step); }; rafRef.current = requestAnimationFrame(step); return () => rafRef.current && cancelAnimationFrame(rafRef.current); }, [value, duration]); const formatted = decimals > 0 ? display.toFixed(decimals) : Math.round(display).toString(); return <>{formatted}{suffix}</>;
};

const MarksUpload = () => {
  // Uploader States (no changes)
  const [file, setFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState([]);
  const [marks, setMarks] = useState([]);
  const [saving, setSaving] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [invalidRows, setInvalidRows] = useState([]);
  
  // Common States
  const [examName, setExamName] = useState("Midterm");
  const [academicYear, setAcademicYear] = useState("2024-25");
  const [err, setErr] = useState("");
  const [infoMsg, setInfoMsg] = useState("");
  const [loading, setLoading] = useState(false);

  // NEW: States for displaying database records
  const [studentReports, setStudentReports] = useState([]);
  const [fetching, setFetching] = useState(false);
  const [fetchErr, setFetchErr] = useState("");

  const navigate = useNavigate();

  // NEW: Function to fetch records from the database
  const fetchStudentReports = async () => {
    if (!examName || !academicYear) {
      setFetchErr("Exam Name and Academic Year are required to fetch records.");
      return;
    }
    setFetching(true);
    setFetchErr("");
    try {
      const apiUrl = import.meta.env.VITE_API_URL;
      const res = await axios.get(`${apiUrl}/api/marks`, {
        params: { exam: examName, academicYear },
      });
      setStudentReports(res.data);
      if(res.data.length === 0) {
        setFetchErr(`No records found for ${examName} (${academicYear}).`);
      }
    } catch (error) {
      setFetchErr(error.response?.data?.message || "Failed to fetch records.");
    } finally {
      setFetching(false);
    }
  };

  // NEW: useEffect to fetch data when the page loads or filters change
  useEffect(() => {
    fetchStudentReports();
  }, []); // Run once on initial component mount

  const handleSave = async () => {
    if (!marks.length || invalidRows.length > 0) {
      return setErr("Cannot save. Correct the issues in the CSV or upload a valid file.");
    }
    try {
      setSaving(true);
      setErr("");
      const apiUrl = import.meta.env.VITE_API_URL;
      const res = await axios.post(`${apiUrl}/api/marks/save-batch`, {
        marks,
        fileName: file?.name || "manual_upload",
      });
      setInfoMsg(`Save successful! Records Created: ${res.data.inserted}, Updated: ${res.data.modified}.`);
      
      // Clear the uploader state after saving
      setMarks([]);
      setCsvPreview([]);
      setFile(null);

      // NEW: Refresh the table on the right with the new data
      fetchStudentReports();

    } catch (error) {
      console.error("Save error:", error);
      setErr(error.response?.data?.message || "Failed to save marks to the database.");
    } finally {
      setSaving(false);
    }
  };

  // NEW: Memoized transformation to flatten the report data for the table
  const flatReportData = useMemo(() => {
    if (!studentReports.length) return [];
    return studentReports.flatMap(report =>
      report.subjects.map(subject => ({
        studentName: report.studentName,
        studentEmail: report._id,
        subject: subject.subject,
        marksObtained: subject.marksObtained,
        outOf: subject.outOf,
        percentage: `${subject.percentage.toFixed(1)}%`
      }))
    );
  }, [studentReports]);
  
  // All other functions (parseCsv, handleFileSelect, etc.) remain unchanged
  const parseCsv = (selected) => { setLoading(true); setErr(""); if (!examName.trim()) { setErr("Please enter an Exam/Term name."); setLoading(false); return; } if (!academicYear.trim()) { setErr("Please enter an Academic Year."); setLoading(false); return; } Papa.parse(selected, { header: true, skipEmptyLines: "greedy", complete: (results) => { try { const rawRows = results.data || []; const cleanedRawRows = rawRows.filter((r) => Object.values(r || {}).some((v) => (v || "").toString().trim() !== "")); setCsvPreview(cleanedRawRows.slice(0, 5)); const allMarks = []; const issues = []; const subjectColumns = ["cns", "dc", "toc", "aisc", "se"]; cleanedRawRows.forEach((row, idx) => { const studentEmail = (row.student_email || "").trim(); const outOf = Number(row.out_of); if (!studentEmail) issues.push({ rowNumber: idx + 1, errors: ["Missing student_email"] }); if (!outOf || outOf <= 0) issues.push({ rowNumber: idx + 1, errors: ["Invalid 'out_of' value"] }); if (!studentEmail || !outOf || outOf <= 0) return; subjectColumns.forEach(subject => { const marksObtained = Number(row[subject]); if (isFinite(marksObtained)) { allMarks.push({ studentEmail, subject: subject.toUpperCase(), marksObtained, outOf, percentage: (marksObtained / outOf) * 100, exam: examName, academicYear, studentName: row.student_name || null, }); } else { issues.push({ rowNumber: idx + 1, errors: [`Invalid marks for subject '${subject.toUpperCase()}'`] }); } }); }); setInvalidRows(issues); setMarks(allMarks); setInfoMsg(`Parsed ${cleanedRawRows.length} students • ${allMarks.length} valid mark entries • ${issues.length} issues found.`); setErr(""); } catch (e) { console.error("Processing error:", e); setErr("Failed to process CSV. Check the file format and columns."); } finally { setLoading(false); } }, error: (error) => { setErr("Failed to parse CSV: " + error.message); setLoading(false); }, }); };
  const handleFileSelect = (selectedFile) => { if (!selectedFile) return; if (!/\.csv$/i.test(selectedFile.name)) { setErr("Please select a .csv file."); return; } setFile(selectedFile); setErr(""); setInfoMsg(""); setMarks([]); setInvalidRows([]); setCsvPreview([]); parseCsv(selectedFile); };
  const handleFileInputChange = (e) => handleFileSelect(e.target.files?.[0]);
  const handleDrop = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); handleFileSelect(e.dataTransfer.files?.[0]); };
  const handleDragEvents = (e) => { e.preventDefault(); e.stopPropagation(); if (e.type === "dragover" || e.type === "dragenter") { setDragActive(true); } else if (e.type === "dragleave" || e.type === "drop") { setDragActive(false); } };
  const downloadTemplate = () => { const headers = "student_email,student_name,cns,dc,toc,aisc,se,out_of"; const sample = 'student1@example.com,"Alex Doe",85,78,92,67,88,100\nstudent2@example.com,"Maria Garcia",72,65,80,75,79,100'; const csv = `${headers}\n${sample}`; const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "marks_template.csv"; a.click(); URL.revokeObjectURL(url); };
  const stats = useMemo(() => { if (!marks.length) return null; const percents = marks.map((m) => m.percentage); const total = percents.length; const avg = total ? percents.reduce((a, b) => a + b, 0) / total : 0; const high = percents.filter((v) => v >= 85).length; const med = percents.filter((v) => v >= 60 && v < 85).length; const low = percents.filter((v) => v < 60).length; const subjectStats = {}; marks.forEach((m) => { if (!subjectStats[m.subject]) { subjectStats[m.subject] = { sum: 0, count: 0 }; } subjectStats[m.subject].sum += m.percentage; subjectStats[m.subject].count += 1; }); Object.keys(subjectStats).forEach(s => { subjectStats[s].avg = subjectStats[s].count > 0 ? subjectStats[s].sum / subjectStats[s].count : 0; }); return { total, avg, high, med, low, subjectStats }; }, [marks]);
  const gaugeColor = stats?.avg >= 85 ? "#28a745" : stats?.avg >= 60 ? "#fd7e14" : "#dc3545";

  // NEW: Updated return statement with Grid layout
  return (
    <Grid container spacing={4}>
      {/* --- LEFT SIDE: UPLOADER --- */}
      <Grid item xs={12} md={7}>
        <div className="marks-upload-container">
          <Card className="upload-card">
            <CardContent>
              <div className="title-row"><Typography variant="h5">Marks Upload & Analysis</Typography><Chip icon={<InsightsIcon />} label="Upload → Analyze → Save" size="small" className="flow-chip"/></div>
              <Divider sx={{ my: 2 }} />
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
                <TextField label="Exam / Term Name" value={examName} onChange={(e) => setExamName(e.target.value)} placeholder="e.g., Midterm 1" fullWidth required/>
                <TextField label="Academic Year" value={academicYear} onChange={(e) => setAcademicYear(e.target.value)} placeholder="e.g., 2024-25" fullWidth required/>
              </Stack>
              <div className={`dropzone ${dragActive ? "drag-active" : ""}`} onDragEnter={handleDragEvents} onDragLeave={handleDragEvents} onDragOver={handleDragEvents} onDrop={handleDrop}>
                <SchoolIcon className="dropzone-icon" />
                <Typography variant="h6" component="p">Drag & drop CSV file here</Typography>
                <Typography color="textSecondary">or</Typography>
                <Stack direction="row" spacing={1} justifyContent="center" sx={{ mt: 1 }}>
                  <Button component="label" variant="outlined" startIcon={<CloudUploadIcon />}>{file ? file.name : "Select File"}<input type="file" accept=".csv" hidden onChange={handleFileInputChange} /></Button>
                  <Tooltip title="Download CSV template with required columns"><Button variant="text" startIcon={<DownloadIcon />} onClick={downloadTemplate}>Template</Button></Tooltip>
                </Stack>
              </div>
              {loading && <LinearProgress sx={{ mt: 2 }} />}
              {err && <Typography color="error" align="center" sx={{ mt: 1 }}>{err}</Typography>}
              {infoMsg && !err && <Typography color="primary" align="center" sx={{ mt: 1 }}>{infoMsg}</Typography>}
              {csvPreview.length > 0 && ( <Box className="csv-preview"> <div className="preview-header"> <Typography variant="subtitle1">File Preview</Typography> {invalidRows.length > 0 && (<Chip color="warning" icon={<WarningAmberRoundedIcon />} label={`${invalidRows.length} issues found`}/>)} </div> <ResultsTable data={csvPreview} compact /> {invalidRows.length > 0 && (<div className="invalid-block"> <Typography variant="subtitle2">Validation Issues</Typography> <ul> {invalidRows.slice(0, 5).map((row, i) => (<li key={i}>Row {row.rowNumber}: {row.errors.join(" • ")}</li>))} {invalidRows.length > 5 && <li>...and {invalidRows.length - 5} more issues.</li>} </ul> </div>)} </Box>)}
            </CardContent>
          </Card>
          {stats && (<Box className="stats-panel fade-in-up"> <Typography variant="h6" className="stats-title">Upload Insights</Typography> <div className="insights-grid"> <div className="gauge-card" style={{"--value": stats.avg, "--gauge-color": gaugeColor}}> <div className="gauge"><div className="gauge-inner"> <div className="gauge-label"> <span className="gauge-value"><CountUp value={stats.avg} decimals={1} suffix="%" /></span> <span className="gauge-sub">Overall Avg</span> </div> </div></div> <div className="gauge-meta"><Chip icon={<SummarizeIcon />} label={`${stats.total} total scores`} size="small"/></div> </div> <div className="levels-grid"> <div className="level-card high"><div className="level-top"><CheckCircleRoundedIcon/><span>≥ 85%</span></div><div className="level-values"><strong><CountUp value={stats.high}/></strong></div></div> <div className="level-card med"><div className="level-top"><WarningAmberRoundedIcon/><span>60-84%</span></div><div className="level-values"><strong><CountUp value={stats.med}/></strong></div></div> <div className="level-card low"><div className="level-top"><ReportRoundedIcon/><span>&lt; 60%</span></div><div className="level-values"><strong><CountUp value={stats.low}/></strong></div></div> </div> <div className="subjects-card"> <Typography variant="subtitle2">Subject Performance</Typography> <div className="subjects-list"> {Object.entries(stats.subjectStats).sort().map(([subj, s]) => { const color = s.avg >= 85 ? "#28a745" : s.avg >= 60 ? "#fd7e14" : "#dc3545"; return ( <div key={subj} className="subject-row"> <span className="subject-name">{subj}</span> <div className="subject-bar"><div className="subject-bar-fill" style={{ width: `${s.avg}%`, background: color }}/></div> <span className="subject-avg">{s.avg.toFixed(1)}%</span> </div> ); })} </div> </div> </div> <Button variant="contained" color="success" onClick={handleSave} disabled={saving || !marks.length || invalidRows.length > 0} startIcon={saving ? <CircularProgress size={20} /> : <SaveIcon />} className="save-btn"> {saving ? "Saving..." : `Save ${marks.length} Mark Entries`} </Button> </Box>)}
        </div>
      </Grid>
      
      {/* --- RIGHT SIDE: DATABASE RECORDS TABLE --- */}
      <Grid item xs={12} md={5}>
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                    <Typography variant="h5">Saved Records</Typography>
                    <Button variant="outlined" size="small" onClick={fetchStudentReports} disabled={fetching}>
                      Refresh
                    </Button>
                </Stack>
                {fetching ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><CircularProgress /></Box>
                ) : fetchErr ? (
                    <Typography color="error" align="center">{fetchErr}</Typography>
                ) : (
                    <ResultsTable data={flatReportData} />
                )}
            </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default MarksUpload;