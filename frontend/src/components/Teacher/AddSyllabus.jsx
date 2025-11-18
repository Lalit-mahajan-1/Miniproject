import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DescriptionIcon from "@mui/icons-material/Description";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import RefreshIcon from "@mui/icons-material/Refresh";
import "./Syllabus.css";

const apiUrl = import.meta.env.VITE_ML_SERVER_URL;

const AddSyllabus = () => {
  const [subjectName, setSubjectName] = useState("");
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [syllabusText, setSyllabusText] = useState("");
  const [syllabusId, setSyllabusId] = useState("");
  const [subjects, setSubjects] = useState([]);
  const [loadingSubjects, setLoadingSubjects] = useState(false);

  const canUpload = useMemo(() => subjectName.trim().length >= 2 && !!file, [subjectName, file]);

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!/(\.pdf|\.png|\.jpg|\.jpeg|\.webp)$/i.test(f.name)) {
      setError("Please upload a PDF or image (png/jpg/jpeg/webp).");
      return;
    }
    setFile(f);
    setError("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    if (!/(\.pdf|\.png|\.jpg|\.jpeg|\.webp)$/i.test(f.name)) {
      setError("Please upload a PDF or image (png/jpg/jpeg/webp).");
      return;
    }
    setFile(f);
    setError("");
  };
  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); };

  const fetchSubjects = async () => {
    try {
      setLoadingSubjects(true);
      const res = await axios.get(`${apiUrl}/ocr/syllabus`);
      setSubjects(res.data?.items || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingSubjects(false);
    }
  };

  useEffect(() => { fetchSubjects(); }, []);

  const handleUpload = async () => {
    if (!canUpload) return;
    try {
      setUploading(true);
      setError("");
      setInfo("");
      setSyllabusText("");
      setSyllabusId("");

      const form = new FormData();
      form.append("subject_name", subjectName.trim());
      form.append("file", file);

      const res = await axios.post(`${apiUrl}/ocr/syllabus`, form, {
        timeout: 120000,
      });

      setSyllabusText(res.data?.text || "");
      setSyllabusId(res.data?.subject_id || "");
      setInfo("Syllabus extracted successfully!");
      fetchSubjects();
    } catch (e) {
      console.error(e);
      setError(e.response?.data?.detail || "Failed to upload or extract syllabus.");
    } finally {
      setUploading(false);
    }
  };

  const loadSubject = async (id) => {
    try {
      setError("");
      setInfo("");
      const res = await axios.get(`${apiUrl}/ocr/syllabus/${id}`);
      setSubjectName(res.data?.name || "");
      setSyllabusText(res.data?.text || "");
      setSyllabusId(id);
    } catch (e) {
      console.error(e);
      setError("Failed to load syllabus.");
    }
  };

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(syllabusText || "");
      setInfo("Copied syllabus text to clipboard.");
      setTimeout(() => setInfo(""), 1500);
    } catch {}
  };

  return (
    <div className="syllabus-page">
      <Card className="syllabus-card">
        <CardContent>
          <div className="title-row">
            <Typography variant="h5" className="title-text">
              Add Syllabus (PDF/Image → OCR)
            </Typography>
            <Chip icon={<DescriptionIcon />} label="PDF or Image supported" size="small" className="flow-chip" />
          </div>

          <Divider className="divider" />

          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems="stretch">
            <Box sx={{ flex: 1 }}>
              <TextField
                fullWidth
                label="Subject Name"
                placeholder="e.g., Data Structures (CSE-201)"
                value={subjectName}
                onChange={(e) => setSubjectName(e.target.value)}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title="Clear">
                        <IconButton size="small" onClick={() => setSubjectName("")}>
                          <RefreshIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </InputAdornment>
                  ),
                }}
              />

              <div
                className={`sy-dropzone ${dragActive ? "drag-active" : ""}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
              >
                <CloudUploadIcon className="sy-icon" />
                <div className="sy-text">
                  <strong>Drag & drop the syllabus file here</strong>
                  <span>PDF or Image (PNG/JPG/JPEG/WebP)</span>
                </div>
                <Button component="label" variant="outlined" className="sy-upload-btn" startIcon={<CloudUploadIcon />}>
                  {file ? file.name : "Browse files"}
                  <input type="file" hidden accept=".pdf,image/*" onChange={onFileChange} />
                </Button>
              </div>

              <Stack direction="row" spacing={2} justifyContent="center">
                <Button
                  variant="contained"
                  className="sy-submit"
                  onClick={handleUpload}
                  disabled={!canUpload || uploading}
                  startIcon={uploading ? <CircularProgress size={18} /> : <DescriptionIcon />}
                >
                  {uploading ? "Processing..." : "Extract Syllabus"}
                </Button>
              </Stack>

              {error && <Typography color="error" align="center" className="sy-msg">{error}</Typography>}
              {info && <Typography color="primary" align="center" className="sy-msg">{info}</Typography>}
            </Box>

            <Box className="sy-sidebar" sx={{ width: { xs: "100%", md: 320 } }}>
              <Typography variant="subtitle1" className="sy-subtitle">Recent Subjects</Typography>
              <Paper className="sy-list">
                {loadingSubjects ? (
                  <Box p={2} display="flex" justifyContent="center"><CircularProgress size={20} /></Box>
                ) : (
                  <List dense>
                    {subjects.length === 0 && <ListItem><ListItemText primary="No uploads yet" /></ListItem>}
                    {subjects.map((s) => (
                      <ListItem key={s.id} button onClick={() => loadSubject(s.id)}>
                        <ListItemText
                          primary={s.name}
                          secondary={new Date(s.created_at).toLocaleString()}
                        />
                      </ListItem>
                    ))}
                  </List>
                )}
              </Paper>
            </Box>
          </Stack>
        </CardContent>
      </Card>

      {syllabusText && (
        <Card className="syllabus-output fade-in-up">
          <CardContent>
            <div className="output-header">
              <Typography variant="h6">Extracted Syllabus</Typography>
              <Stack direction="row" spacing={1}>
                <Chip label={subjectName || "Unnamed"} size="small" />
                <Tooltip title="Copy to clipboard">
                  <IconButton size="small" onClick={copyText}><ContentCopyIcon fontSize="small" /></IconButton>
                </Tooltip>
              </Stack>
            </div>
            <Paper className="output-box">
              <pre className="syllabus-pre">{syllabusText}</pre>
            </Paper>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AddSyllabus;