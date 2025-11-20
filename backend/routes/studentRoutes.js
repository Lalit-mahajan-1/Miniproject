const express = require("express");
const router = express.Router();
const studentController = require("../controllers/studentController");

// GET /api/students - Get all students
router.get("/", studentController.getAllStudents);

// POST /api/students - Create single student
router.post("/", studentController.createStudent);

// POST /api/students/bulk - Create multiple students
router.post("/bulk", studentController.createStudentsBulk);

// GET /api/students/analytics - Get analytics data
router.get("/analytics", studentController.getAnalytics);

// GET /api/students/:id - Get student by ID
router.get("/:id", studentController.getStudentById);

// PUT /api/students/:id - Update student
router.put("/:id", studentController.updateStudent);

// DELETE /api/students/:id - Delete student
router.delete("/:id", studentController.deleteStudent);

// POST /api/students/mentor-request - Student requests a mentor
router.post("/mentor-request", async (req, res) => {
    try {
        const userId = req.body.userId; // In frontend we will send this

        if (!userId) {
            return res.status(400).json({ message: "User ID is required" });
        }

        const MentorRequest = require("../models/MentorRequest");
        const User = require("../models/User");

        // Check if user exists
        const student = await User.findById(userId);
        if (!student || student.role !== "student") {
            return res.status(404).json({ message: "Student not found" });
        }

        // Check if student already has a pending request
        const existing = await MentorRequest.findOne({
            student: userId,
            status: "pending",
        });

        if (existing) {
            return res.json({ status: "pending", message: "Request already pending" });
        }

        // Create new mentor request
        await MentorRequest.create({
            student: userId,
            status: "pending",
        });

        res.json({ status: "success", message: "Mentor request submitted" });
    } catch (err) {
        console.error("MENTOR REQUEST ERROR:", err);
        res.status(500).json({ message: "Failed to create mentor request" });
    }
});

// GET /api/student/mentor-status/:userId
router.get("/mentor-status/:userId", async (req, res) => {
    try {
        const { userId } = req.params;

        const MentorRequest = require("../models/MentorRequest");

        // Find latest request for this student
        const request = await MentorRequest.findOne({ student: userId }).populate("assignedTeacher", "name email").sort({ createdAt: -1 });

        if (!request) {
            return res.json({ status: "none" }); // no request yet
        }

        if (request.status === "pending") {
            return res.json({ status: "pending" });
        }

        if (request.status === "approved") {
            return res.json({
                status: "approved",
                mentor: request.assignedTeacher,
            });
        }

        return res.json({ status: request.status });
    } catch (err) {
        console.error("MENTOR STATUS ERROR:", err);
        res.status(500).json({ message: "Failed to get mentor status" });
    }
});

module.exports = router;
