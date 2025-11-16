const express = require("express");
const router = express.Router();
const marksController = require("../controllers/marks.controller");
// Optional: Add authentication middleware here if you have one
// const { protect } = require("../middleware/authMiddleware");

// POST /api/marks/save-batch
// Saves a batch of marks sent in the request body.
router.post("/save-batch", marksController.saveBatch);

// GET /api/marks
// Fetches mark records, can be filtered by query params like ?batchId=...
router.get("/", marksController.getMarks);
// Add this new route for fetching a student's own marks
router.get("/me", marksController.getMyLatestMarks);

module.exports = router;