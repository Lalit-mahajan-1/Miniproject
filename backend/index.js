// backend/server.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const { connectDB } = require("./config/db");

const authRoutes = require("./routes/auth");
const studentRiskRoutes = require("./routes/studentRisks");
const marksRoutes = require("./routes/marks.routes");

// NEW ROUTES (Mentor System)
const adminRoutes = require("./routes/adminRoutes");
const studentRoutes = require("./routes/studentRoutes");
const teacherRoutes = require("./routes/teacherRoutes");
// const chatbotRoutes = require("./routes/chatbot.routes");
const chatbotRoutes = require("./routes/chatbot.routes");


const app = express();
const PORT = process.env.PORT || 5000;

// -------------------
// CORS CONFIG
// -------------------
const allowedOrigins = (process.env.CLIENT_ORIGIN || "").split(",").filter(Boolean);

app.use(
    cors({
        origin: allowedOrigins.length > 0 ? allowedOrigins : "*",
        credentials: true,
    })
);

// -------------------
// MIDDLEWARE
// -------------------
app.use(express.json({ limit: "10mb" }));
app.use(morgan("dev"));

// -------------------
// HEALTH CHECK
// -------------------
app.get("/api/health", (req, res) => {
    res.json({ status: "ok", message: "API is healthy" });
});

// -------------------
// ROUTES
// -------------------
app.use("/api/auth", authRoutes);
app.use("/api/student-risks", studentRiskRoutes);
app.use("/api/marks", marksRoutes);

// Mentor management routes
app.use("/api/admin", adminRoutes);

// Student actions (mentor request)
app.use("/api/student", studentRoutes);

app.use("/api/teacher", teacherRoutes);

// app.use("/api", chatbotRoutes);
app.use("/api", chatbotRoutes);



// -------------------
// ERROR HANDLER
// -------------------
app.use((err, req, res, next) => {
    console.error("Unhandled API Error:", err.stack);
    res.status(500).json({ message: "Internal Server Error" });
});

// -------------------
// START SERVER
// -------------------
connectDB(process.env.MONGO_URI)
    .then(() => {
        app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
    })
    .catch((e) => {
        console.error("❌ Failed to connect to MongoDB", e);
        process.exit(1);
    });
