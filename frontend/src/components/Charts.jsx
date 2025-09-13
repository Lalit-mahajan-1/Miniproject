import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const Charts = ({ studentData }) => {
  // Grade distribution data
  const gradeDistribution = studentData.reduce((acc, student) => {
    const avg = Math.round((parseInt(student.math) + parseInt(student.science) + parseInt(student.english)) / 3);
    let grade = avg >= 90 ? 'A' : avg >= 80 ? 'B' : avg >= 70 ? 'C' : avg >= 60 ? 'D' : 'F';
    acc[grade] = (acc[grade] || 0) + 1;
    return acc;
  }, {});

  const gradeData = Object.entries(gradeDistribution).map(([grade, count]) => ({
    grade, count, percentage: Math.round((count / studentData.length) * 100)
  }));

  // Risk distribution data
  const riskData = studentData.reduce((acc, student) => {
    acc[student.risk] = (acc[student.risk] || 0) + 1;
    return acc;
  }, {});

  const riskChartData = Object.entries(riskData).map(([level, count]) => ({
    level, count, percentage: Math.round((count / studentData.length) * 100)
  }));

  // Subject performance data
  const subjectData = ['math', 'science', 'english'].map(subject => ({
    subject: subject.charAt(0).toUpperCase() + subject.slice(1),
    average: Math.round(studentData.reduce((sum, s) => sum + parseInt(s[subject]), 0) / studentData.length),
    passing: studentData.filter(s => parseInt(s[subject]) >= 60).length,
  }));

  const COLORS = {
    High: '#ef4444',
    Medium: '#f59e0b', 
    Low: '#10b981',
    A: '#10b981',
    B: '#3b82f6',
    C: '#f59e0b',
    D: '#f97316',
    F: '#ef4444'
  };

  const PIE_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Grade Distribution */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
          Grade Distribution
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={gradeData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              dataKey="count"
              label={({grade, percentage}) => `${grade}: ${percentage}%`}
            >
              {gradeData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[entry.grade] || PIE_COLORS[index]} />
              ))}
            </Pie>
            <Tooltip formatter={(value, name) => [value, 'Students']} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Risk Level Distribution */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
          Risk Levels
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={riskChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis dataKey="level" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip 
              formatter={(value) => [value, 'Students']}
              labelStyle={{ color: '#374151' }}
              contentStyle={{ backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Bar 
              dataKey="count" 
              fill={(entry) => COLORS[entry.level]}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Subject Performance */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 lg:col-span-2">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
          Subject Performance
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={subjectData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis dataKey="subject" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip 
              formatter={(value, name) => [
                name === 'average' ? `${value}%` : `${value} students`, 
                name === 'average' ? 'Average Score' : 'Passing Students'
              ]}
              labelStyle={{ color: '#374151' }}
              contentStyle={{ backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '8px' }}
            />
            <Bar dataKey="average" fill="#3b82f6" name="average" radius={[4, 4, 0, 0]} />
            <Bar dataKey="passing" fill="#10b981" name="passing" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Charts;