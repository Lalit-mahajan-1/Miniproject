// backend/routes/teacherRoutes.js
const express = require("express");
const MentorRequest = require("../models/MentorRequest");
const User = require("../models/User");
const router = express.Router();

// --------------------------------------
// GET: Students assigned to teacher
// --------------------------------------
router.get("/mentees", async (req, res) => {
    try {
        const teacherId = req.query.teacherId;

        if (!teacherId) {
            return res.status(400).json({
                success: false,
                message: "teacherId is required",
            });
        }

        const mentees = await MentorRequest.find({
            assignedTeacher: teacherId,
            status: "approved",
        })
            .populate("student", "name email department year rollNo phone")
            .lean();

        res.json({ success: true, data: mentees });
    } catch (err) {
        console.error("Mentorship fetch error:", err);
        res.status(500).json({ success: false, message: "Server Error" });
    }
});

// --------------------------------------
// DELETE: Remove a student from mentorship
// --------------------------------------
// POST: Remove a student from mentorship (Delete database entry)
router.post("/remove", async (req, res) => {
    try {
        const { requestId } = req.body;

        if (!requestId) {
            return res.status(400).json({
                success: false,
                message: "requestId is required",
            });
        }

        const deleted = await MentorRequest.findByIdAndDelete(requestId);

        if (!deleted) {
            return res.status(404).json({
                success: false,
                message: "Mentor request not found",
            });
        }

        res.json({
            success: true,
            message: "Student removed from mentorship and record deleted.",
        });
    } catch (err) {
        console.error("Mentorship delete error:", err);
        res.status(500).json({
            success: false,
            message: "Server Error",
        });
    }
});


module.exports = router;
