const Student = require('../models/Student');

// Get all students
const getAllStudents = async (req, res) => {
  try {
    const students = await Student.find().sort({ createdAt: -1 });
    res.json({ success: true, data: students, count: students.length });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
};

// Create single student
const createStudent = async (req, res) => {
  try {
    const student = new Student(req.body);
    await student.save();
    res.status(201).json({ success: true, data: student });
  } catch (error) {
    res.status(400).json({ success: false, message: error.message });
  }
};

// Create multiple students
const createStudentsBulk = async (req, res) => {
  try {
    const { students } = req.body;
    const createdStudents = await Student.insertMany(students);
    res.status(201).json({ 
      success: true, 
      data: createdStudents, 
      count: createdStudents.length 
    });
  } catch (error) {
    res.status(400).json({ success: false, message: error.message });
  }
};

// Get student by ID
const getStudentById = async (req, res) => {
  try {
    const student = await Student.findById(req.params.id);
    if (!student) {
      return res.status(404).json({ success: false, message: 'Student not found' });
    }
    res.json({ success: true, data: student });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
};

// Update student
const updateStudent = async (req, res) => {
  try {
    const student = await Student.findByIdAndUpdate(
      req.params.id, 
      req.body, 
      { new: true, runValidators: true }
    );
    if (!student) {
      return res.status(404).json({ success: false, message: 'Student not found' });
    }
    res.json({ success: true, data: student });
  } catch (error) {
    res.status(400).json({ success: false, message: error.message });
  }
};

// Delete student
const deleteStudent = async (req, res) => {
  try {
    const student = await Student.findByIdAndDelete(req.params.id);
    if (!student) {
      return res.status(404).json({ success: false, message: 'Student not found' });
    }
    res.json({ success: true, message: 'Student deleted successfully' });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
};

// Get analytics data
const getAnalytics = async (req, res) => {
  try {
    const students = await Student.find();
    
    const analytics = {
      totalStudents: students.length,
      riskDistribution: {
        High: students.filter(s => s.risk === 'High').length,
        Medium: students.filter(s => s.risk === 'Medium').length,
        Low: students.filter(s => s.risk === 'Low').length
      },
      averageScores: {
        math: Math.round(students.reduce((sum, s) => sum + s.math, 0) / students.length) || 0,
        science: Math.round(students.reduce((sum, s) => sum + s.science, 0) / students.length) || 0,
        english: Math.round(students.reduce((sum, s) => sum + s.english, 0) / students.length) || 0,
        attendance: Math.round(students.reduce((sum, s) => sum + s.attendance, 0) / students.length) || 0
      }
    };
    
    res.json({ success: true, data: analytics });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
};

module.exports = {
  getAllStudents,
  createStudent,
  createStudentsBulk,
  getStudentById,
  updateStudent,
  deleteStudent,
  getAnalytics
};