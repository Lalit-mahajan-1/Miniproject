const mongoose = require('mongoose');

const studentSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, 'Student name is required'],
    trim: true,
    maxlength: [100, 'Name cannot exceed 100 characters']
  },
  math: {
    type: Number,
    required: [true, 'Math score is required'],
    min: [0, 'Score cannot be negative'],
    max: [100, 'Score cannot exceed 100']
  },
  science: {
    type: Number,
    required: [true, 'Science score is required'],
    min: [0, 'Score cannot be negative'],
    max: [100, 'Score cannot exceed 100']
  },
  english: {
    type: Number,
    required: [true, 'English score is required'],
    min: [0, 'Score cannot be negative'],
    max: [100, 'Score cannot exceed 100']
  },
  attendance: {
    type: Number,
    required: [true, 'Attendance is required'],
    min: [0, 'Attendance cannot be negative'],
    max: [100, 'Attendance cannot exceed 100']
  },
  risk: {
    type: String,
    enum: ['Low', 'Medium', 'High'],
    default: 'Low'
  }
}, {
  timestamps: true
});

// Calculate risk level before saving
studentSchema.pre('save', function(next) {
  const avgGrade = (this.math + this.science + this.english) / 3;
  
  if (avgGrade < 60 || this.attendance < 75) {
    this.risk = 'High';
  } else if (avgGrade < 75 || this.attendance < 85) {
    this.risk = 'Medium';
  } else {
    this.risk = 'Low';
  }
  
  next();
});

// Virtual for average grade
studentSchema.virtual('averageGrade').get(function() {
  return Math.round((this.math + this.science + this.english) / 3);
});

// Include virtuals in JSON
studentSchema.set('toJSON', { virtuals: true });

module.exports = mongoose.model('Student', studentSchema);