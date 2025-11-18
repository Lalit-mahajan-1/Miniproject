import axios from 'axios'

// Base URLs for different services
const API_BASE_URL = '/api'
const ML_BASE_URL = '/ml'

// Create axios instances
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

const mlClient = axios.create({
  baseURL: ML_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// API functions for student data
export const studentAPI = {
  // Get all students
  getAllStudents: async () => {
    try {
      const response = await apiClient.get('/students')
      return response.data
    } catch (error) {
      console.error('Error fetching students:', error)
      throw error
    }
  },

  // Add a single student
  addStudent: async (studentData) => {
    try {
      const response = await apiClient.post('/students', studentData)
      return response.data
    } catch (error) {
      console.error('Error adding student:', error)
      throw error
    }
  },

  // Add multiple students from CSV
  addStudentsBulk: async (studentsData) => {
    try {
      const response = await apiClient.post('/students/bulk', { students: studentsData })
      return response.data
    } catch (error) {
      console.error('Error adding students in bulk:', error)
      throw error
    }
  },

  // Update a student
  updateStudent: async (studentId, studentData) => {
    try {
      const response = await apiClient.put(`/students/${studentId}`, studentData)
      return response.data
    } catch (error) {
      console.error('Error updating student:', error)
      throw error
    }
  },

  // Delete a student
  deleteStudent: async (studentId) => {
    try {
      const response = await apiClient.delete(`/students/${studentId}`)
      return response.data
    } catch (error) {
      console.error('Error deleting student:', error)
      throw error
    }
  },
}

// ML API functions
export const mlAPI = {
  // Get risk prediction for a student
  predictRisk: async (studentData) => {
    try {
      const response = await mlClient.post('/predict-risk', studentData)
      return response.data
    } catch (error) {
      console.error('Error predicting risk:', error)
      throw error
    }
  },

  // Get risk predictions for multiple students
  predictRiskBulk: async (studentsData) => {
    try {
      const response = await mlClient.post('/predict-risk-bulk', { students: studentsData })
      return response.data
    } catch (error) {
      console.error('Error predicting risk for multiple students:', error)
      throw error
    }
  },

  // Get analytics insights
  getAnalytics: async (studentsData) => {
    try {
      const response = await mlClient.post('/analytics', { students: studentsData })
      return response.data
    } catch (error) {
      console.error('Error fetching analytics:', error)
      throw error
    }
  },
}

// Utility functions
export const utils = {
  // Calculate grade average
  calculateGradeAverage: (student) => {
    const grades = [
      parseFloat(student.math) || 0,
      parseFloat(student.science) || 0,
      parseFloat(student.english) || 0,
    ]
    return Math.round(grades.reduce((sum, grade) => sum + grade, 0) / grades.length)
  },

  // Calculate risk level based on grades and attendance
  calculateRisk: (student) => {
    const avgGrade = utils.calculateGradeAverage(student)
    const attendance = parseFloat(student.attendance) || 0

    if (avgGrade < 60 || attendance < 75) return 'High'
    if (avgGrade < 75 || attendance < 85) return 'Medium'
    return 'Low'
  },

  // Get grade letter from average
  getGradeLetter: (average) => {
    if (average >= 90) return 'A'
    if (average >= 80) return 'B'
    if (average >= 70) return 'C'
    if (average >= 60) return 'D'
    return 'F'
  },

  // Format percentage
  formatPercentage: (value, total) => {
    if (total === 0) return '0%'
    return `${Math.round((value / total) * 100)}%`
  },

  // Validate CSV data structure
  validateCSVData: (data) => {
    const requiredFields = ['name', 'math', 'science', 'english', 'attendance']
    const errors = []

    if (!Array.isArray(data) || data.length === 0) {
      errors.push('No valid data found')
      return { isValid: false, errors }
    }

    data.forEach((row, index) => {
      const missingFields = requiredFields.filter(field => !row.hasOwnProperty(field) || row[field] === '')
      if (missingFields.length > 0) {
        errors.push(`Row ${index + 1}: Missing fields - ${missingFields.join(', ')}`)
      }

      // Validate numeric fields
      const numericFields = ['math', 'science', 'english', 'attendance']
      numericFields.forEach(field => {
        if (row[field] && (isNaN(row[field]) || row[field] < 0 || row[field] > 100)) {
          errors.push(`Row ${index + 1}: ${field} must be a number between 0 and 100`)
        }
      })
    })

    return {
      isValid: errors.length === 0,
      errors
    }
  },
}

// Error handler for API calls
export const handleAPIError = (error, defaultMessage = 'An error occurred') => {
  if (error.response) {
    // Server responded with error status
    return error.response.data?.message || error.response.statusText || defaultMessage
  } else if (error.request) {
    // Request was made but no response received
    return 'No response from server. Please check your connection.'
  } else {
    // Something else happened
    return error.message || defaultMessage
  }
}

export default {
  studentAPI,
  mlAPI,
  utils,
  handleAPIError,
}