const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const studentRoutes = require('./routes/studentRoutes');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/teacher_dashboard')
  .then(() => console.log('✅ MongoDB Connected'))
  .catch(err => console.error('❌ MongoDB Error:', err));

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Teacher Dashboard API is running' });
});

app.use('/api/students', studentRoutes);

// Error handler
app.use((error, req, res, next) => {
  console.error(error);
  res.status(500).json({ 
    success: false, 
    message: error.message || 'Server Error' 
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});