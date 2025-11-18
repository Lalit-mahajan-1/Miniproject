import React, { useState } from 'react';
import { UserPlus, Save, RefreshCw, CheckCircle } from 'lucide-react';

const CustomInput = ({ addStudent }) => {
  const [student, setStudent] = useState({
    name: '',
    math: '',
    science: '',
    english: '',
    attendance: ''
  });
  const [errors, setErrors] = useState({});
  const [showSuccess, setShowSuccess] = useState(false);

  const validateForm = () => {
    const newErrors = {};
    
    if (!student.name.trim()) newErrors.name = 'Name is required';
    if (!student.math || student.math < 0 || student.math > 100) newErrors.math = 'Math score must be 0-100';
    if (!student.science || student.science < 0 || student.science > 100) newErrors.science = 'Science score must be 0-100';
    if (!student.english || student.english < 0 || student.english > 100) newErrors.english = 'English score must be 0-100';
    if (!student.attendance || student.attendance < 0 || student.attendance > 100) newErrors.attendance = 'Attendance must be 0-100';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      addStudent(student);
      setStudent({ name: '', math: '', science: '', english: '', attendance: '' });
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    }
  };

  const handleReset = () => {
    setStudent({ name: '', math: '', science: '', english: '', attendance: '' });
    setErrors({});
  };

  const InputField = ({ label, name, type = "text", placeholder, icon: Icon }) => (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
      <div className="relative">
        {Icon && <Icon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />}
        <input
          type={type}
          name={name}
          value={student[name]}
          onChange={(e) => setStudent({ ...student, [name]: e.target.value })}
          placeholder={placeholder}
          className={`w-full ${Icon ? 'pl-10' : 'pl-3'} pr-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
            errors[name] ? 'border-red-500 bg-red-50' : 'border-gray-300'
          }`}
        />
      </div>
      {errors[name] && <p className="text-red-500 text-xs mt-1">{errors[name]}</p>}
    </div>
  );

  const avg = student.math && student.science && student.english 
    ? Math.round((parseInt(student.math) + parseInt(student.science) + parseInt(student.english)) / 3)
    : 0;

  const riskLevel = avg < 60 || parseInt(student.attendance) < 75 ? 'High' : 
                   avg < 75 || parseInt(student.attendance) < 85 ? 'Medium' : 'Low';

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
          <div className="flex items-center mb-2">
            <UserPlus className="text-blue-600 mr-3" size={24} />
            <h1 className="text-2xl font-bold text-gray-800">Add New Student</h1>
          </div>
          <p className="text-gray-600">Enter student information manually to add to the dashboard.</p>
        </div>

        {/* Success Message */}
        {showSuccess && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center animate-fade-in">
            <CheckCircle className="text-green-600 mr-3" size={20} />
            <span className="text-green-800 font-medium">Student added successfully!</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 space-y-6">
          {/* Student Name */}
          <InputField
            label="Student Name"
            name="name"
            placeholder="Enter student's full name"
          />

          {/* Grades Section */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
              Subject Scores
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <InputField
                label="Mathematics"
                name="math"
                type="number"
                placeholder="0-100"
              />
              <InputField
                label="Science"
                name="science"
                type="number"
                placeholder="0-100"
              />
              <InputField
                label="English"
                name="english"
                type="number"
                placeholder="0-100"
              />
            </div>
          </div>

          {/* Attendance */}
          <InputField
            label="Attendance Percentage"
            name="attendance"
            type="number"
            placeholder="0-100"
          />

          {/* Preview */}
          {student.name && student.math && student.science && student.english && student.attendance && (
            <div className="bg-gray-50 rounded-lg p-4 border">
              <h4 className="font-semibold text-gray-800 mb-3">Preview</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Name:</span>
                  <p className="font-medium">{student.name}</p>
                </div>
                <div>
                  <span className="text-gray-600">Average Grade:</span>
                  <p className="font-medium">{avg}%</p>
                </div>
                <div>
                  <span className="text-gray-600">Attendance:</span>
                  <p className="font-medium">{student.attendance}%</p>
                </div>
                <div>
                  <span className="text-gray-600">Risk Level:</span>
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    riskLevel === 'High' ? 'bg-red-100 text-red-800' :
                    riskLevel === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {riskLevel}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 pt-4">
            <button
              type="submit"
              className="flex-1 flex items-center justify-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Save className="mr-2" size={16} />
              Add Student
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="flex-1 flex items-center justify-center px-6 py-3 bg-gray-200 text-gray-800 font-medium rounded-lg hover:bg-gray-300 transition-colors"
            >
              <RefreshCw className="mr-2" size={16} />
              Reset Form
            </button>
          </div>
        </form>

        {/* Tips */}
        <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
          <h3 className="font-semibold text-blue-800 mb-2">Tips for Data Entry</h3>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Ensure all grades are between 0-100</li>
            <li>• Attendance should be entered as a percentage (0-100)</li>
            <li>• Double-check student names for accuracy</li>
            <li>• Risk levels are automatically calculated based on grades and attendance</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default CustomInput;