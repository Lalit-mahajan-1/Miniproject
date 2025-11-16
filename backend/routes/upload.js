// backend/routes/upload.js
const express = require('express');
const User = require('../models/User.js');
const { auth, authorize } = require('../middleware/auth');

const router = express.Router();

// Update student risk data in database (optional route)
router.post('/update-student-risks', auth, authorize('teacher'), async (req, res) => {
  try {
    const { students } = req.body;
    
    // Update each student's risk data
    for (const record of students) {
      await User.findOneAndUpdate(
        { studentId: record.student_id, role: 'student' },
        {
          riskData: {
            predicted_risk_percentage: record.predicted_risk_percentage,
            risk_category: record.risk_category,
            predicted_score: record.predicted_score,
            last_updated: new Date()
          }
        }
      );
    }

    res.json({ message: 'Student risk data updated successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Update failed', error: error.message });
  }
});

// Get student risk data (for students to view their own)
router.get('/student/risk', auth, authorize('student'), async (req, res) => {
  try {
    const student = await User.findById(req.user.id).select('riskData studentId name');
    res.json(student);
  } catch (error) {
    res.status(500).json({ message: 'Server error' });
  }
});

// Get all students risk data (for teachers and admin)
router.get('/students/risk', auth, authorize('teacher', 'admin'), async (req, res) => {
  try {
    const students = await User.find({ role: 'student' })
      .select('name email studentId riskData')
      .sort({ 'riskData.predicted_risk_percentage': -1 });
    res.json(students);
  } catch (error) {
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;