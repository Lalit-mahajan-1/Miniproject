// backend/routes/adminRoutes.js
const express = require("express");
const router = express.Router();
const User = require("../models/User");
const MentorRequest = require("../models/MentorRequest");
// const { requireAdmin } = require("../middleware/auth"); // make sure you have this

// Get pending mentor requests
router.get(
    "/mentor-requests",
    /*requireAdmin,*/ async (req, res) => {
        try {
            const requests = await MentorRequest.find({ status: "pending" }).populate("student", "name email studentId").sort({ createdAt: -1 });

            res.json(requests);
        } catch (err) {
            console.error(err);
            res.status(500).json({ message: "Error fetching mentor requests" });
        }
    }
);

// Assign mentor to student + mark request approved
// Assign mentor to student + mark request approved
router.post("/assign-mentor", async (req, res) => {
    try {
        const { studentId, teacherId, requestId } = req.body;

        if (!studentId || !teacherId) {
            return res.status(400).json({ message: "studentId and teacherId are required" });
        }

        const student = await User.findById(studentId);
        const teacher = await User.findById(teacherId);

        if (!student || !teacher) {
            return res.status(404).json({ message: "Student or teacher not found" });
        }

        if (teacher.role !== "teacher") {
            return res.status(400).json({ message: "Selected mentor is not a teacher" });
        }

        // Assign mentor to student schema
        student.mentor = teacher._id;
        await student.save();

        // Update mentor request with selected teacher
        if (requestId) {
            await MentorRequest.findByIdAndUpdate(requestId, {
                status: "approved",
                assignedTeacher: teacher._id, // IMPORTANT
            });
        }

        res.json({ message: "Mentor assigned successfully", student });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Error assigning mentor" });
    }
});

// Get all teachers and their students
// Get all teachers and their students
router.get("/mentors-map", async (req, res) => {
    try {
        // Get all teachers first
        const teachers = await User.find({ role: "teacher" }).select("name email");

        // Get all approved mentor assignments from MentorRequest
        const approved = await MentorRequest.find({
            status: "approved",
            assignedTeacher: { $ne: null },
        })
            .populate("assignedTeacher", "name email")
            .populate("student", "name email studentId");

        // Build mapping: teacher → students[]
        const result = teachers.map((teacher) => ({
            teacher,
            students: approved.filter((req) => req.assignedTeacher?._id.toString() === teacher._id.toString()).map((req) => req.student), // Only return student object
        }));

        res.json(result);
    } catch (err) {
        console.error("MENTORS-MAP ERROR:", err);
        res.status(500).json({ message: "Error fetching mentors map" });
    }
});

// Get count of students assigned to a specific teacher
router.get("/mentor-count/:teacherId", async (req, res) => {
    try {
        const { teacherId } = req.params;

        const count = await MentorRequest.countDocuments({
            assignedTeacher: teacherId,
            status: "approved",
        });

        res.json({ teacherId, assignedCount: count });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Error fetching mentor count" });
    }
});

module.exports = router;
