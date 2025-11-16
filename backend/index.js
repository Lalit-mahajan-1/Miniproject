// backend/server.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const { connectDB } = require("./config/db");

// Import Routes
const authRoutes = require('./routes/auth');
const studentRiskRoutes = require("./routes/studentRisks");
const marksRoutes = require("./routes/marks.routes"); // <-- NEW: Import marks routes

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware Setup
const allowedOrigins = (process.env.CLIENT_ORIGIN || "").split(",").filter(Boolean);
app.use(
  cors({
    origin: allowedOrigins.length > 0 ? allowedOrigins : "*",
    credentials: true,
  })
);
app.use(express.json({ limit: "10mb" }));
app.use(morgan("dev"));

// API Health Check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "API is healthy" });
});

// API Routes
app.use('/api/auth', authRoutes);
app.use("/api/student-risks", studentRiskRoutes);
app.use("/api/marks", marksRoutes); // <-- NEW: Use the marks routes

// Global Error Handler (must be last)
app.use((err, req, res, next) => {
  console.error("Unhandled API Error:", err.stack);
  res.status(500).json({ message: "Internal Server Error" });
});

// Connect to DB and Start Server
connectDB(process.env.MONGO_URI)
  .then(() => {
    app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
  })
  .catch((e) => {
    console.error("❌ Failed to connect to MongoDB", e);
    process.exit(1);
  });