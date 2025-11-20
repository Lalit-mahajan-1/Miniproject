const express = require("express");
const router = express.Router();

const { GoogleGenerativeAI } = require("@google/generative-ai");

// Init Gemini Model
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

router.post("/chat", async (req, res) => {
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({ success: false, message: "query is required" });
    }

    const model = genAI.getGenerativeModel({ model: "models/gemini-2.5-flash" });

    const result = await model.generateContent(query);

    return res.json({
      success: true,
      response: result.response.text(),
    });

  } catch (error) {
    console.error("🔥 GEMINI ERROR:", error);
    return res.status(500).json({
      success: false,
      message: "Gemini API failed to generate response",
      error: error.message,
    });
  }
});

module.exports = router;
