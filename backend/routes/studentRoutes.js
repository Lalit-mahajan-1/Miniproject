const express = require('express');
const router = express.Router();
const studentController = require('../controllers/studentController');

// GET /api/students - Get all students
router.get('/', studentController.getAllStudents);

// POST /api/students - Create single student
router.post('/', studentController.createStudent);

// POST /api/students/bulk - Create multiple students
router.post('/bulk', studentController.createStudentsBulk);

// GET /api/students/analytics - Get analytics data
router.get('/analytics', studentController.getAnalytics);

// GET /api/students/:id - Get student by ID
router.get('/:id', studentController.getStudentById);

// PUT /api/students/:id - Update student
router.put('/:id', studentController.updateStudent);

// DELETE /api/students/:id - Delete student
router.delete('/:id', studentController.deleteStudent);

module.exports = router;