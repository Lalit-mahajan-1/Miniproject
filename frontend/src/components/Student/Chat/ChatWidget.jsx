// frontend/src/Components/Student/Chat/ChatWidget.jsx
import { useEffect, useState, useRef } from "react";
import axios from "axios";
import {
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  TextField,
  Button,
  CircularProgress,
  Typography,
  Stack,
  Paper,
  Switch,
  FormControlLabel,
  Tooltip,
  Alert,
} from "@mui/material";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import SendRoundedIcon from "@mui/icons-material/SendRounded";
import SchoolRoundedIcon from "@mui/icons-material/SchoolRounded";
import ForumRoundedIcon from "@mui/icons-material/ForumRounded";

// Main component for the chat dialog
const ChatWidget = ({ open, onClose, studentContext }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // RAG (Syllabus) vs General (Mental Health, etc.)
  const [useRag, setUseRag] = useState(true);

  const chatEndRef = useRef(null);
  // Use ML server URL for chatbot (FastAPI on port 8000)
  const mlServerUrl = import.meta.env.VITE_ML_SERVER_URL;

  // Validate ML server URL
  useEffect(() => {
    if (!mlServerUrl) {
      console.error("❌ VITE_ML_SERVER_URL is not set in environment variables!");
      setError("Configuration error: ML Server URL not found. Please contact support.");
    }
  }, [mlServerUrl]);

  // Scroll to bottom when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Set initial welcome message when dialog opens
  useEffect(() => {
    if (open) {
      setMessages([
        {
          sender: "bot",
          text: "Hi there! 👋 I'm your AI study assistant.\n\n• Toggle the switch below to ask about your syllabus (RAG mode) or chat generally about stress, study tips, etc.\n• Currently in: " + (useRag ? "Syllabus Mode 📚" : "General Chat Mode 💬"),
        },
      ]);
      setError("");
    }
  }, [open, useRag]);

  const handleSend = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) {
      setError("Please type a message before sending.");
      return;
    }

    if (!mlServerUrl) {
      setError("ML Server URL is not configured. Please check your environment settings.");
      return;
    }

    const userMessage = { sender: "user", text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError("");

    // Create FormData to match backend expectations
    const formData = new FormData();
    formData.append("query", userMessage.text);
    formData.append("use_rag", useRag.toString());

    try {
      console.log(`📤 Sending chat request to: ${mlServerUrl}/chat/`);
      console.log(`   Query: "${userMessage.text}"`);
      console.log(`   Mode: ${useRag ? "RAG (Syllabus)" : "General Chat"}`);

      const res = await axios.post(`${mlServerUrl}/chat/`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 30000, // 30 second timeout
      });

      console.log("✅ Response received:", res.data);

      const botMessage = {
        sender: "bot",
        text: res.data?.response || "I'm not sure how to respond to that. Could you rephrase your question?",
      };
      setMessages((prev) => [...prev, botMessage]);
      
    } catch (err) {
      console.error("❌ Chat API error:", err);
      
      let errorMsg = "An unexpected error occurred. Please try again.";
      
      if (err.code === "ECONNABORTED") {
        errorMsg = "Request timed out. The server might be busy. Please try again.";
      } else if (err.response) {
        // Server responded with error
        errorMsg = err.response.data?.detail || 
                   err.response.data?.message || 
                   `Server error (${err.response.status})`;
        console.error("Server response:", err.response.data);
      } else if (err.request) {
        // Request made but no response
        errorMsg = "Cannot reach the server. Please check if the backend is running.";
        console.error("No response from server");
      }
      
      setError(errorMsg);
      setMessages((prev) => [
        ...prev,
        { 
          sender: "bot", 
          text: `⚠️ ${errorMsg}\n\nIf this persists, please contact your teacher or system administrator.` 
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleModeSwitch = (e) => {
    const newMode = e.target.checked;
    setUseRag(newMode);
    
    // Add a system message when mode changes
    const modeMessage = {
      sender: "system",
      text: `Switched to ${newMode ? "Syllabus Mode 📚" : "General Chat Mode 💬"}`,
    };
    setMessages((prev) => [...prev, modeMessage]);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="sm"
      PaperProps={{ 
        sx: { 
          height: "80vh", 
          maxHeight: "700px",
          display: "flex",
          flexDirection: "column"
        } 
      }}
    >
      <DialogTitle
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          pb: 1,
          borderBottom: "1px solid #e5e7eb"
        }}
      >
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            AI Study Assistant
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {useRag ? "📚 Syllabus Mode" : "💬 General Chat Mode"}
          </Typography>
        </Box>
        <IconButton onClick={onClose} aria-label="close">
          <CloseRoundedIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent 
        dividers 
        sx={{ 
          p: 0, 
          bgcolor: "#f9fafb",
          flex: 1,
          overflow: "hidden"
        }}
      >
        <Stack sx={{ height: "100%", p: 2, overflowY: "auto" }}>
          {messages.map((msg, index) => (
            <ChatMessage 
              key={index} 
              sender={msg.sender} 
              text={msg.text} 
            />
          ))}
          {loading && (
            <Box sx={{ display: "flex", justifyContent: "flex-start", mb: 1 }}>
              <Paper
                elevation={0}
                sx={{
                  px: 1.5,
                  py: 1,
                  bgcolor: "#e5e7eb",
                  borderRadius: "12px 12px 12px 0",
                  display: "flex",
                  alignItems: "center",
                  gap: 1
                }}
              >
                <CircularProgress size={16} />
                <Typography variant="caption">Thinking...</Typography>
              </Paper>
            </Box>
          )}
          <div ref={chatEndRef} />
        </Stack>
      </DialogContent>

      <DialogActions sx={{ p: 1.5, display: "block", bgcolor: "#fff" }}>
        {error && (
          <Alert severity="error" sx={{ mb: 1.5 }} onClose={() => setError("")}>
            {error}
          </Alert>
        )}
        
        <Stack direction="row" spacing={1} component="form" onSubmit={handleSend}>
          <TextField
            fullWidth
            size="small"
            variant="outlined"
            placeholder={useRag ? "Ask about your syllabus..." : "How can I help you today?"}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            autoFocus
          />
          <Button
            type="submit"
            variant="contained"
            disabled={loading || !input.trim()}
            sx={{ px: 2.5, minWidth: "auto" }}
          >
            <SendRoundedIcon />
          </Button>
        </Stack>
        
        <Stack
          direction="row"
          justifyContent="center"
          alignItems="center"
          sx={{ mt: 1.5 }}
        >
          <Tooltip title="General Chat / Mental Health Support">
            <ForumRoundedIcon
              fontSize="small"
              color={!useRag ? "primary" : "disabled"}
            />
          </Tooltip>
          <Switch
            size="small"
            checked={useRag}
            onChange={handleModeSwitch}
            inputProps={{ "aria-label": "Chat mode toggle" }}
          />
          <Tooltip title="Syllabus Questions (RAG)">
            <SchoolRoundedIcon
              fontSize="small"
              color={useRag ? "primary" : "disabled"}
            />
          </Tooltip>
        </Stack>
      </DialogActions>
    </Dialog>
  );
};

// Helper component to render individual messages
const ChatMessage = ({ sender, text }) => {
  const isUser = sender === "user";
  const isSystem = sender === "system";
  
  if (isSystem) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", my: 1 }}>
        <Typography 
          variant="caption" 
          sx={{ 
            bgcolor: "#f3f4f6", 
            px: 2, 
            py: 0.5, 
            borderRadius: 2,
            color: "text.secondary"
          }}
        >
          {text}
        </Typography>
      </Box>
    );
  }
  
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        mb: 1.5,
      }}
    >
      <Paper
        elevation={0}
        sx={{
          px: 1.5,
          py: 1,
          maxWidth: "80%",
          bgcolor: isUser ? "primary.main" : "#e5e7eb",
          color: isUser ? "primary.contrastText" : "text.primary",
          borderRadius: isUser
            ? "12px 12px 0 12px"
            : "12px 12px 12px 0",
          whiteSpace: "pre-wrap",
          wordWrap: "break-word",
        }}
      >
        <Typography variant="body2">{text}</Typography>
      </Paper>
    </Box>
  );
};

export default ChatWidget;