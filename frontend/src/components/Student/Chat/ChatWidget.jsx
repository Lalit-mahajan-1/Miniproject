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
  Tooltip,
  Alert,
  Zoom,
} from "@mui/material";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import SendRoundedIcon from "@mui/icons-material/SendRounded";
import SchoolRoundedIcon from "@mui/icons-material/SchoolRounded";
import ForumRoundedIcon from "@mui/icons-material/ForumRounded";
import SmartToyIcon from '@mui/icons-material/SmartToy';

const ChatWidget = ({ open, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  // Default to Syllabus Mode (RAG)
  const [useRag, setUseRag] = useState(true);

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // 1. Define Server URL with a fallback to localhost:8000
  const ML_SERVER_URL = import.meta.env.VITE_ML_SERVER_URL || "http://localhost:8000";
  // const ML_SERVER_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  // 2. Scroll to bottom when messages change
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // 3. Focus input when opened
  useEffect(() => {
    if (open) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [open]);

  // 4. Set initial welcome message (Check length to prevent React StrictMode duplication)
  useEffect(() => {
    if (open && messages.length === 0) {
      setMessages([
        {
          sender: "bot",
          text: "Hi there! 👋 I'm your AI study assistant.\n\n• Toggle the switch below to ask about your **Syllabus** (RAG mode) or chat generally about **Mental Health & Study Tips**.",
        },
      ]);
    }
  }, [open]);

  const handleSend = async (e) => {
    e.preventDefault();

    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError("");

    // Create FormData for FastAPI Form(...)
    const formData = new FormData();
    formData.append("query", userMessage.text);
    formData.append("use_rag", useRag.toString()); // Sends "true" or "false"

    try {
      // 5. Axios Call
      const formData = new FormData();
formData.append("query", userMessage.text);

const res = await axios.post(
  `${ML_SERVER_URL}/chat/gemini`,
  formData,
  {
    headers: { "Content-Type": "multipart/form-data" }
  }
);



      if (res.data && res.data.success) {
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: res.data.response },
        ]);
      } else {
        throw new Error("Invalid response format");
      }

    } catch (err) {
      console.error("Chat Error:", err);
      const errMsg = err.response?.data?.detail || "Could not connect to the AI server.";
      
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: `⚠️ Error: ${errMsg}` },
      ]);
    } finally {
      setLoading(false);
      // Keep focus on input after sending
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleModeSwitch = (e) => {
    const newMode = e.target.checked;
    setUseRag(newMode);
    setMessages((prev) => [
      ...prev,
      {
        sender: "system",
        text: `Switched to ${newMode ? "📚 Syllabus Mode" : "💬 General Chat Mode"}`,
      },
    ]);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="sm"
      TransitionComponent={Zoom} // Nice open animation
      PaperProps={{
        sx: {
          height: "80vh",
          maxHeight: "700px",
          borderRadius: 3,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden"
        },
      }}
    >
      {/* Header */}
      <DialogTitle
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          bgcolor: "primary.main",
          color: "white",
          py: 1.5,
        }}
      >
        <Stack direction="row" alignItems="center" spacing={1}>
          <SmartToyIcon />
          <Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
              AI Assistant
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.9 }}>
              {useRag ? "Searching Syllabus Documents" : "General Support Chat"}
            </Typography>
          </Box>
        </Stack>
        <IconButton onClick={onClose} sx={{ color: "white" }}>
          <CloseRoundedIcon />
        </IconButton>
      </DialogTitle>

      {/* Chat Area */}
      <DialogContent
        sx={{
          p: 2,
          bgcolor: "#f4f6f8",
          flex: 1,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Stack spacing={2}>
          {messages.map((msg, index) => (
            <ChatMessage key={index} sender={msg.sender} text={msg.text} />
          ))}

          {loading && (
            <Box sx={{ display: "flex", justifyContent: "flex-start" }}>
              <Paper
                sx={{
                  px: 2,
                  py: 1.5,
                  bgcolor: "white",
                  borderRadius: "18px 18px 18px 0",
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <CircularProgress size={14} />
                <Typography variant="caption" color="text.secondary">
                  Thinking...
                </Typography>
              </Paper>
            </Box>
          )}
          <div ref={chatEndRef} />
        </Stack>
      </DialogContent>

      {/* Input Area */}
      <DialogActions
        sx={{
          p: 2,
          bgcolor: "white",
          borderTop: "1px solid #e0e0e0",
          display: "flex",
          flexDirection: "column",
          gap: 1
        }}
      >
        {/* Toggle Switch */}
        <Stack
          direction="row"
          alignItems="center"
          spacing={1}
          sx={{ width: "100%", justifyContent: "center", mb: 1 }}
        >
          <Tooltip title="General Chat / Mental Health">
            <ForumRoundedIcon
              fontSize="small"
              color={!useRag ? "primary" : "action"}
            />
          </Tooltip>
          <Switch
            checked={useRag}
            onChange={handleModeSwitch}
            size="small"
          />
          <Tooltip title="Ask Syllabus Questions">
            <SchoolRoundedIcon
              fontSize="small"
              color={useRag ? "primary" : "action"}
            />
          </Tooltip>
          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
            {useRag ? "Syllabus Mode" : "General Mode"}
          </Typography>
        </Stack>

        {/* Input Field */}
        <Stack
          component="form"
          onSubmit={handleSend}
          direction="row"
          spacing={1}
          sx={{ width: "100%" }}
        >
          <TextField
            inputRef={inputRef}
            fullWidth
            size="small"
            placeholder={
              useRag
                ? "e.g., 'What are the modules in Python?'"
                : "e.g., 'I feel stressed about exams'"
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            sx={{ 
              "& .MuiOutlinedInput-root": { borderRadius: 3 } 
            }}
          />
          <Button
            type="submit"
            variant="contained"
            disabled={loading || !input.trim()}
            sx={{ 
              borderRadius: 3, 
              minWidth: "50px", 
              height: "40px" 
            }}
          >
            <SendRoundedIcon fontSize="small" />
          </Button>
        </Stack>
      </DialogActions>
    </Dialog>
  );
};

// --- Helper Message Component ---
const ChatMessage = ({ sender, text }) => {
  const isUser = sender === "user";
  const isSystem = sender === "system";

  if (isSystem) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", opacity: 0.8 }}>
        <Typography variant="caption" sx={{ color: "text.secondary", fontStyle: "italic" }}>
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
      }}
    >
      <Paper
        elevation={isUser ? 4 : 1}
        sx={{
          px: 2,
          py: 1.5,
          maxWidth: "75%",
          bgcolor: isUser ? "primary.main" : "white",
          color: isUser ? "white" : "text.primary",
          borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
          position: "relative",
        }}
      >
        <Typography
          variant="body2"
          sx={{
            whiteSpace: "pre-wrap", // Preserves line breaks from bot
            wordBreak: "break-word",
          }}
        >
            {/* Handle simplified markdown (bolding) */}
            {text.split("**").map((part, i) => 
              i % 2 === 1 ? <strong key={i}>{part}</strong> : part
            )}
        </Typography>
      </Paper>
    </Box>
  );
};

export default ChatWidget;