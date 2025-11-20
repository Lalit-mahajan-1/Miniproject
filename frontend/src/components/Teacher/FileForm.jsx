import { useState, useMemo, useEffect, useRef } from "react";
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
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import SaveIcon from "@mui/icons-material/Save";
import InsightsIcon from "@mui/icons-material/Insights";
import ReportRoundedIcon from "@mui/icons-material/ReportRounded";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import ResultsTable from "./ResultsTable";
import Papa from "papaparse";

const CountUp = ({ value = 0, decimals = 0, duration = 1200, suffix = "" }) => {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef(null);

  useEffect(() => {
    let start;
    const from = 0;
    const to = Number(value) || 0;
    const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = easeOutCubic(progress);
      const current = from + (to - from) * eased;
      setDisplay(current);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      }
    };

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [value, duration]);

  const formatted =
    decimals > 0 ? display.toFixed(decimals) : Math.round(display).toString();

  return (
    <>
      {formatted}
      {suffix}
    </>
  );
};

const FileForm = () => {
  const [file, setFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState([]);
  const [data, setData] = useState([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [infoMsg, setInfoMsg] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const navigate = useNavigate();

  const parsePreview = (selected) => {
    Papa.parse(selected, {
      header: true,
      skipEmptyLines: "greedy",
      complete: (results) => {
        const rows = (results.data || []).filter((r) =>
          Object.values(r || {}).some((v) => (v || "").toString().trim() !== "")
        );
        setCsvPreview(rows.slice(0, 5)); // first 5 rows
      },
      error: (error) => setErr("Failed to parse CSV: " + error.message),
    });
  };

  const handleFileInputChange = (e) => {
    const selected = e.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setErr("");
    setInfoMsg("");
    setData([]);
    parsePreview(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (!dropped) return;
    if (!/\.csv$/i.test(dropped.name)) {
      setErr("Please drop a .csv file.");
      return;
    }
    setFile(dropped);
    setErr("");
    setInfoMsg("");
    setData([]);
    parsePreview(dropped);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handlePredict = async () => {
    if (!file) return setErr("Please select a CSV file first!");

    const formData = new FormData();
    formData.append("file_upload", file);

    try {
      setLoading(true);
      setErr("");
      setInfoMsg("");

      const base =
        import.meta.env.VITE_ML_SERVER_URL ||
        import.meta.env.VITE_API_URL ||
        window.location.origin;
      const mlUrl = `${base}`.replace(/\/+$/, ""); // strip trailing slash

      const mlRes = await axios.post(`${mlUrl}/uploadfile`, formData, {
        // Do not set Content-Type; browser sets proper multipart boundary
        timeout: 60000,
      });

      const predictedData = mlRes.data?.data || [];
      setData(predictedData);
      setInfoMsg("Prediction completed successfully!");
    } catch (error) {
      console.error("Prediction error:", error);
      let msg =
        error?.response?.data?.detail ||
        error?.response?.statusText ||
        error?.message ||
        "Prediction failed. Try again.";
      // Friendlier message for server-down cases
      if (error?.code === "ERR_NETWORK") {
        const base =
          import.meta.env.VITE_ML_SERVER_URL ||
          import.meta.env.VITE_API_URL ||
          window.location.origin;
        msg = `Cannot reach ML server at ${base}/uploadfile. Is it running?`;
      }
      setErr(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!data || data.length === 0)
      return setErr("No predictions to save. Upload & Predict first!");
    try {
      setSaving(true);
      const apiUrl =
        import.meta.env.VITE_API_URL || window.location.origin;
      const res = await axios.post(`${apiUrl.replace(/\/+$/, "")}/api/student-risks/save-batch`, {
        students: data,
        fileName: file?.name || undefined,
      });
      navigate(`/teacher/records?batchId=${res.data?.batchId}`);
    } catch (error) {
      console.error("Save error:", error);
      setErr(
        error?.response?.data?.detail ||
          error?.response?.statusText ||
          "Failed to save results."
      );
    } finally {
      setSaving(false);
    }
  };

  const stats = useMemo(() => {
    if (!data.length) return null;
    const risks = data.map((r) => Number(r.predicted_risk_percentage || 0));
    const total = risks.length;
    const avg = risks.reduce((a, b) => a + b, 0) / total;
    const high = risks.filter((v) => v > 70).length;
    const med = risks.filter((v) => v > 40 && v <= 70).length;
    const low = risks.filter((v) => v <= 40).length;
    const highPct = (high / total) * 100;
    const medPct = (med / total) * 100;
    const lowPct = (low / total) * 100;

    return {
      totalStudents: total,
      avg,
      avgLabel: avg.toFixed(1) + "%",
      high,
      med,
      low,
      highPct,
      medPct,
      lowPct,
    };
  }, [data]);

  const gaugeColor =
    stats?.avg >= 70 ? "#dc3545" : stats?.avg > 40 ? "#fd7e14" : "#28a745";

  return (
    <div className="fileform-container">
      <Card className="upload-card">
        <CardContent>
          <div className="title-row">
            <Typography variant="h5" className="title-text">
              Student Risk Prediction
            </Typography>
            <Chip
              icon={<InsightsIcon />}
              label="Upload → Predict → Review"
              size="small"
              className="flow-chip"
            />
          </div>

          <Divider className="divider" />

          {/* Drag & Drop Zone */}
          <div
            className={`dropzone ${dragActive ? "drag-active" : ""}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <CloudUploadIcon className="dropzone-icon" />
            <div className="dropzone-text">
              <strong>Drag & drop your CSV here</strong>
              <span>or click to browse</span>
            </div>
            <Button
              component="label"
              variant="outlined"
              startIcon={<CloudUploadIcon />}
              className="upload-btn"
            >
              {file ? file.name : "Select CSV File"}
              <input
                type="file"
                accept=".csv"
                hidden
                onChange={handleFileInputChange}
              />
            </Button>
          </div>

          <Stack
            direction="row"
            justifyContent="center"
            spacing={2}
            className="btn-group"
          >
            <Button
              variant="contained"
              color="primary"
              onClick={handlePredict}
              disabled={loading || !file}
              startIcon={
                loading ? <CircularProgress size={20} /> : <AnalyticsIcon />
              }
              className="predict-btn"
            >
              {loading ? "Predicting..." : "Predict"}
            </Button>
          </Stack>

          {loading && <LinearProgress className="progress-bar" />}

          {err && (
            <Typography color="error" align="center" className="message">
              {err}
            </Typography>
          )}
          {infoMsg && !err && (
            <Typography color="primary" align="center" className="message">
              {infoMsg}
            </Typography>
          )}

          {/* CSV Preview */}
          {csvPreview.length > 0 && (
            <div className="csv-preview">
              <Typography variant="subtitle1" className="preview-title">
                File Preview (First 5 Rows)
              </Typography>
              <ResultsTable data={csvPreview} compact />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Enhanced Stats after Prediction */}
      {stats && (
        <Box className="stats-panel fade-in-up">
          <Typography variant="h6" className="stats-title">
            Prediction Insights
          </Typography>

          <div className="insights-grid">
            <div
              className="gauge-card"
              style={{
                "--value": stats.avg,
                "--gauge-color": gaugeColor,
              }}
            >
              <div className="gauge">
                <div className="gauge-inner">
                  <div className="gauge-label">
                    <span className="gauge-value">
                      <CountUp value={stats.avg} decimals={1} suffix="%" />
                    </span>
                    <span className="gauge-sub">Avg Risk</span>
                  </div>
                </div>
              </div>
              <div className="gauge-meta">
                <Chip label={`${stats.totalStudents} students`} size="small" />
              </div>
            </div>

            <div className="levels-grid">
              <div className="level-card high">
                <div className="level-top">
                  <ReportRoundedIcon />
                  <span>High Risk</span>
                </div>
                <div className="level-values">
                  <strong>
                    <CountUp value={stats.high} />
                  </strong>
                  <span>
                    <CountUp value={stats.highPct} decimals={1} suffix="%" />
                  </span>
                </div>
              </div>

              <div className="level-card med">
                <div className="level-top">
                  <WarningAmberRoundedIcon />
                  <span>Medium Risk</span>
                </div>
                <div className="level-values">
                  <strong>
                    <CountUp value={stats.med} />
                  </strong>
                  <span>
                    <CountUp value={stats.medPct} decimals={1} suffix="%" />
                  </span>
                </div>
              </div>

              <div className="level-card low">
                <div className="level-top">
                  <CheckCircleRoundedIcon />
                  <span>Low Risk</span>
                </div>
                <div className="level-values">
                  <strong>
                    <CountUp value={stats.low} />
                  </strong>
                  <span>
                    <CountUp value={stats.lowPct} decimals={1} suffix="%" />
                  </span>
                </div>
              </div>
            </div>

            <div className="distribution-card">
              <div className="distribution-bar">
                <div
                  className="seg low"
                  style={{ width: `${stats.lowPct}%` }}
                  title={`Low: ${stats.low} (${stats.lowPct.toFixed(1)}%)`}
                />
                <div
                  className="seg med"
                  style={{ width: `${stats.medPct}%` }}
                  title={`Medium: ${stats.med} (${stats.medPct.toFixed(1)}%)`}
                />
                <div
                  className="seg high"
                  style={{ width: `${stats.highPct}%` }}
                  title={`High: ${stats.high} (${stats.highPct.toFixed(1)}%)`}
                />
              </div>
              <div className="distribution-legend">
                <span className="legend-item low-dot">
                  Low • {stats.low} ({stats.lowPct.toFixed(1)}%)
                </span>
                <span className="legend-item med-dot">
                  Medium • {stats.med} ({stats.medPct.toFixed(1)}%)
                </span>
                <span className="legend-item high-dot">
                  High • {stats.high} ({stats.highPct.toFixed(1)}%)
                </span>
              </div>
            </div>
          </div>

          <Button
            variant="contained"
            color="success"
            onClick={handleSave}
            disabled={saving}
            startIcon={saving ? <CircularProgress size={20} /> : <SaveIcon />}
            className="save-btn"
          >
            {saving ? "Saving..." : "Save Results"}
          </Button>
        </Box>
      )}

      {data.length > 0 && (
        <div className="results-table-container">
          <Typography variant="h6" className="results-title">
            Prediction Results
          </Typography>
          <ResultsTable data={data} />
        </div>
      )}
    </div>
  );
};

export default FileForm;