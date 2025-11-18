import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams, Link as RouterLink } from "react-router-dom";
import axios from "axios";
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  IconButton,
  InputAdornment,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from "@mui/material";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import DownloadRoundedIcon from "@mui/icons-material/DownloadRounded";
import InsightsIcon from "@mui/icons-material/Insights";
import ReportRoundedIcon from "@mui/icons-material/ReportRounded";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";

const apiUrl = import.meta.env.VITE_API_URL;

const riskColor = (riskCategory) => {
  if (!riskCategory) return { color: "default", hex: "#9ca3af" };
  const txt = String(riskCategory).toLowerCase();
  if (txt.includes("high")) return { color: "error", hex: "#ef4444" };
  if (txt.includes("medium")) return { color: "warning", hex: "#f59e0b" };
  return { color: "success", hex: "#10b981" };
};

const CircularGauge = ({ value = 0, colorHex = "#2563eb", size = 140 }) => {
  const safeVal = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <Box sx={{ position: "relative", display: "inline-flex" }}>
      <CircularProgress variant="determinate" value={100} size={size} thickness={5} sx={{ color: "#e5e7eb" }} />
      <CircularProgress
        variant="determinate"
        value={safeVal}
        size={size}
        thickness={5}
        sx={{ color: colorHex, position: "absolute", left: 0 }}
      />
      <Box
        sx={{
          top: 0,
          left: 0,
          bottom: 0,
          right: 0,
          position: "absolute",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 0.5,
        }}
      >
        <Typography variant="h4" sx={{ fontWeight: 700, lineHeight: 1 }}>
          {safeVal.toFixed(1)}%
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Avg Risk
        </Typography>
      </Box>
    </Box>
  );
};

const StatPill = ({ icon, label, value, pct, color }) => (
  <Paper
    sx={{
      p: 1.5,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      borderRadius: 2,
      bgcolor: "#f8fafc",
    }}
    elevation={0}
  >
    <Stack direction="row" alignItems="center" spacing={1}>
      {icon}
      <Typography variant="body2">{label}</Typography>
    </Stack>
    <Stack alignItems="flex-end" spacing={0}>
      <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
        {value}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {pct?.toFixed ? `${pct.toFixed(1)}%` : pct}
      </Typography>
    </Stack>
  </Paper>
);

function formatDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return String(iso);
  }
}

function toCsv(rows, headers) {
  const esc = (v) => {
    const s = v == null ? "" : String(v);
    // Escape quotes and wrap in quotes if needed
    const needsQuotes = /[",\n]/.test(s);
    const escaped = s.replace(/"/g, '""');
    return needsQuotes ? `"${escaped}"` : escaped;
  };
  const head = headers.map(esc).join(",");
  const body = rows.map((r) => headers.map((h) => esc(r[h])).join(",")).join("\n");
  return `${head}\n${body}`;
}

const Records = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const batchId = searchParams.get("batchId");

  const [loading, setLoading] = useState(false);
  const [loadingBatches, setLoadingBatches] = useState(false);
  const [err, setErr] = useState("");

  const [batch, setBatch] = useState(null);
  const [students, setStudents] = useState([]);
  const [batches, setBatches] = useState([]);
  const [q, setQ] = useState("");

  const loadBatch = async (id) => {
    if (!id) return;
    try {
      setLoading(true);
      setErr("");
      const res = await axios.get(`${apiUrl}/api/student-risks/batch/${id}`, { timeout: 20000 });
      setBatch(res.data?.batch || null);
      setStudents(res.data?.students || []);
    } catch (e) {
      console.error(e);
      setErr(e.response?.data?.message || "Failed to load batch.");
      setBatch(null);
      setStudents([]);
    } finally {
      setLoading(false);
    }
  };

  const loadBatches = async () => {
    try {
      setLoadingBatches(true);
      const res = await axios.get(`${apiUrl}/api/student-risks/batches`, { timeout: 20000 });
      setBatches(res.data?.items || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingBatches(false);
    }
  };

  useEffect(() => {
    loadBatches();
  }, []);

  useEffect(() => {
    if (batchId) loadBatch(batchId);
  }, [batchId]);

  const stats = useMemo(() => {
    if (batch && typeof batch.avg === "number") {
      // Use saved summary from DB
      return {
        totalStudents: batch.totalStudents || 0,
        avg: Number(batch.avg || 0),
        high: Number(batch.high || 0),
        med: Number(batch.med || 0),
        low: Number(batch.low || 0),
        highPct: Number(batch.highPct || 0),
        medPct: Number(batch.medPct || 0),
        lowPct: Number(batch.lowPct || 0),
        createdAt: batch.createdAt,
        fileName: batch.fileName,
      };
    }
    // Fallback: compute from students (shouldn't be needed if backend stored summary)
    const risks = students.map((r) => Number(r.predicted_risk_percentage || 0));
    const total = risks.length || 0;
    const avg = total ? risks.reduce((a, b) => a + b, 0) / total : 0;
    const high = risks.filter((v) => v > 70).length;
    const med = risks.filter((v) => v > 40 && v <= 70).length;
    const low = risks.filter((v) => v <= 40).length;
    const highPct = total ? (high / total) * 100 : 0;
    const medPct = total ? (med / total) * 100 : 0;
    const lowPct = total ? (low / total) * 100 : 0;
    return {
      totalStudents: total,
      avg,
      high,
      med,
      low,
      highPct,
      medPct,
      lowPct,
      createdAt: students[0]?.createdAt,
    };
  }, [batch, students]);

  const gaugeColorHex = stats?.avg >= 70 ? "#ef4444" : stats?.avg > 40 ? "#f59e0b" : "#10b981";

  const filtered = useMemo(() => {
    const text = q.trim().toLowerCase();
    if (!text) return students;
    return students.filter((s) => {
      const email = (s.student_email || "").toLowerCase();
      const name = (s.student_name || "").toLowerCase();
      const cat = (s.risk_category || "").toLowerCase();
      return email.includes(text) || name.includes(text) || cat.includes(text);
    });
  }, [q, students]);

  const exportCsv = () => {
    const headers = [
      "student_email",
      "student_name",
      "student_id",
      "predicted_risk_percentage",
      "risk_category",
      "attendance_percentage",
      "weekly_self_study_hours",
      "class_participation",
      "grade",
      "predicted_score",
      "createdAt",
    ];
    const rows = filtered.map((s) => {
      const out = {};
      for (const h of headers) out[h] = s[h];
      return out;
    });
    const csv = toCsv(rows, headers);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const niceName = stats?.fileName ? stats.fileName.replace(/\.[^.]+$/, "") : "results";
    a.download = `${niceName || "batch"}-${batch?._id || "export"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const goToBatch = (id) => {
    navigate(`/teacher/records?batchId=${id}`);
  };

  return (
    <Box sx={{ px: { xs: 2, md: 3 }, py: 3, bgcolor: "#f8fafc", minHeight: "calc(100vh - 64px)" }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
        <Stack direction="row" alignItems="center" spacing={1.5}>
          <Tooltip title="Back to Upload">
            <span>
              <Button
                variant="outlined"
                size="small"
                component={RouterLink}
                to="/teacher/home"
                startIcon={<ArrowBackRoundedIcon />}
              >
                Upload
              </Button>
            </span>
          </Tooltip>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Saved Prediction Records
          </Typography>
          <Chip icon={<InsightsIcon />} label="Batches" size="small" />
        </Stack>

        <Stack direction="row" spacing={1}>
          <TextField
            size="small"
            placeholder="Search by email, name or category"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchRoundedIcon fontSize="small" />
                </InputAdornment>
              ),
            }}
          />
          <Button
            variant="outlined"
            onClick={() => (batchId ? loadBatch(batchId) : loadBatches())}
            startIcon={<RefreshRoundedIcon />}
          >
            Refresh
          </Button>
          {batchId && students.length > 0 && (
            <Button variant="contained" color="primary" onClick={exportCsv} startIcon={<DownloadRoundedIcon />}>
              Export CSV
            </Button>
          )}
        </Stack>
      </Stack>

      <Grid container spacing={2}>
        {/* Left: Batch detail or empty state */}
        <Grid item xs={12} md={8.5}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              {!batchId ? (
                <Typography variant="body1">
                  Select a batch from the right panel to view its saved results.
                </Typography>
              ) : loading ? (
                <>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>
                    Loading batch…
                  </Typography>
                  <LinearProgress />
                </>
              ) : err ? (
                <Typography color="error">{err}</Typography>
              ) : !batch ? (
                <Typography variant="body2">No batch found for the provided batchId.</Typography>
              ) : (
                <>
                  <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems="center">
                    <CircularGauge value={Number(stats?.avg || 0)} colorHex={gaugeColorHex} size={160} />

                    <Box sx={{ flex: 1, width: "100%" }}>
                      <Grid container spacing={1.5}>
                        <Grid item xs={12} sm={4}>
                          <StatPill
                            icon={<ReportRoundedIcon color="error" />}
                            label="High Risk"
                            value={stats?.high || 0}
                            pct={stats?.highPct || 0}
                          />
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <StatPill
                            icon={<WarningAmberRoundedIcon color="warning" />}
                            label="Medium Risk"
                            value={stats?.med || 0}
                            pct={stats?.medPct || 0}
                          />
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <StatPill
                            icon={<CheckCircleRoundedIcon color="success" />}
                            label="Low Risk"
                            value={stats?.low || 0}
                            pct={stats?.lowPct || 0}
                          />
                        </Grid>
                      </Grid>

                      <Divider sx={{ my: 1.5 }} />

                      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                        <Chip label={`${stats?.totalStudents || 0} students`} size="small" />
                        {stats?.fileName && <Chip label={`File: ${stats.fileName}`} size="small" />}
                        <Chip label={`Created: ${formatDate(stats?.createdAt)}`} size="small" />
                        {batch?.createdBy && <Chip label={`By: ${batch.createdBy}`} size="small" />}
                      </Stack>

                      {/* Distribution bar */}
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          Distribution
                        </Typography>
                        <Box sx={{ display: "flex", height: 12, borderRadius: 8, overflow: "hidden", mt: 0.5 }}>
                          <Box
                            sx={{
                              width: `${stats?.lowPct || 0}%`,
                              bgcolor: "#10b981",
                              transition: "width .3s",
                            }}
                            title={`Low: ${stats?.low} (${(stats?.lowPct || 0).toFixed(1)}%)`}
                          />
                          <Box
                            sx={{
                              width: `${stats?.medPct || 0}%`,
                              bgcolor: "#f59e0b",
                              transition: "width .3s",
                            }}
                            title={`Medium: ${stats?.med} (${(stats?.medPct || 0).toFixed(1)}%)`}
                          />
                          <Box
                            sx={{
                              width: `${stats?.highPct || 0}%`,
                              bgcolor: "#ef4444",
                              transition: "width .3s",
                            }}
                            title={`High: ${stats?.high} (${(stats?.highPct || 0).toFixed(1)}%)`}
                          />
                        </Box>
                      </Box>
                    </Box>
                  </Stack>

                  <Divider sx={{ my: 2 }} />

                  {/* Students table */}
                  <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>
                    Prediction Results
                  </Typography>
                  <Paper sx={{ width: "100%", overflow: "auto" }} elevation={0}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Email</TableCell>
                          <TableCell>Name</TableCell>
                          <TableCell>Student ID</TableCell>
                          <TableCell align="right">Risk %</TableCell>
                          <TableCell>Category</TableCell>
                          <TableCell align="right">Attendance</TableCell>
                          <TableCell align="right">Self Study</TableCell>
                          <TableCell>Participation</TableCell>
                          <TableCell>Grade</TableCell>
                          <TableCell align="right">Predicted Score</TableCell>
                          <TableCell>Saved At</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {filtered.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={11}>
                              <Typography variant="body2" color="text.secondary">
                                {q ? "No matches for your search." : "No students found in this batch."}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ) : (
                          filtered.map((s) => {
                            const catColor = riskColor(s.risk_category);
                            return (
                              <TableRow key={s._id}>
                                <TableCell>{s.student_email}</TableCell>
                                <TableCell>{s.student_name || "—"}</TableCell>
                                <TableCell>{s.student_id || "—"}</TableCell>
                                <TableCell align="right">
                                  {Number(s.predicted_risk_percentage || 0).toFixed(1)}%
                                </TableCell>
                                <TableCell>
                                  <Chip label={s.risk_category} color={catColor.color} size="small" />
                                </TableCell>
                                <TableCell align="right">
                                  {s.attendance_percentage != null ? `${s.attendance_percentage}%` : "—"}
                                </TableCell>
                                <TableCell align="right">
                                  {s.weekly_self_study_hours != null ? s.weekly_self_study_hours : "—"}
                                </TableCell>
                                <TableCell>{s.class_participation || "—"}</TableCell>
                                <TableCell>{s.grade || "—"}</TableCell>
                                <TableCell align="right">
                                  {s.predicted_score != null ? s.predicted_score : "—"}
                                </TableCell>
                                <TableCell>{formatDate(s.createdAt)}</TableCell>
                              </TableRow>
                            );
                          })
                        )}
                      </TableBody>
                    </Table>
                  </Paper>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Right: batches list */}
        <Grid item xs={12} md={3.5}>
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                  Recent Batches
                </Typography>
                <IconButton size="small" onClick={loadBatches}>
                  <RefreshRoundedIcon fontSize="small" />
                </IconButton>
              </Stack>
              <Divider sx={{ mb: 1 }} />
              {loadingBatches ? (
                <Box p={2} display="flex" justifyContent="center">
                  <CircularProgress size={20} />
                </Box>
              ) : (
                <Paper variant="outlined" sx={{ maxHeight: 420, overflowY: "auto" }}>
                  <List dense>
                    {batches.length === 0 && (
                      <ListItem>
                        <ListItemText primary="No batches yet" />
                      </ListItem>
                    )}
                    {batches.map((b) => (
                      <ListItem
                        key={b._id}
                        button
                        selected={b._id === batchId}
                        onClick={() => goToBatch(b._id)}
                        sx={{
                          "&.Mui-selected": {
                            bgcolor: "action.selected",
                          },
                        }}
                      >
                        <ListItemText
                          primary={`Avg: ${Number(b.avg || 0).toFixed(1)}% • ${b.totalStudents} students`}
                          secondary={formatDate(b.createdAt)}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Paper>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Records;