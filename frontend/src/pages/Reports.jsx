import React, { useState } from 'react';
import { BarChart3, Download, Filter, Calendar, Users, TrendingUp } from 'lucide-react';
import Charts from '../components/Charts';

const Reports = ({ studentData }) => {
  const [filterRisk, setFilterRisk] = useState('All');
  const [sortBy, setSortBy] = useState('name');

  const filteredData = studentData.filter(student => 
    filterRisk === 'All' || student.risk === filterRisk
  );

  const sortedData = [...filteredData].sort((a, b) => {
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    if (sortBy === 'average') {
      const avgA = (parseInt(a.math) + parseInt(a.science) + parseInt(a.english)) / 3;
      const avgB = (parseInt(b.math) + parseInt(b.science) + parseInt(b.english)) / 3;
      return avgB - avgA;
    }
    if (sortBy === 'attendance') return parseInt(b.attendance) - parseInt(a.attendance);
    return 0;
  });

  const generateReport = () => {
    const reportData = [
      'Student Name,Math,Science,English,Average,Attendance,Risk Level',
      ...sortedData.map(s => {
        const avg = Math.round((parseInt(s.math) + parseInt(s.science) + parseInt(s.english)) / 3);
        return `${s.name},${s.math},${s.science},${s.english},${avg},${s.attendance},${s.risk}`;
      })
    ].join('\n');

    const blob = new Blob([reportData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `student_report_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const stats = {
    total: studentData.length,
    highRisk: studentData.filter(s => s.risk === 'High').length,
    mediumRisk: studentData.filter(s => s.risk === 'Medium').length,
    lowRisk: studentData.filter(s => s.risk === 'Low').length,
    avgGrade: Math.round(studentData.reduce((sum, s) => {
      const avg = (parseInt(s.math) + parseInt(s.science) + parseInt(s.english)) / 3;
      return sum + avg;
    }, 0) / studentData.length) || 0,
    avgAttendance: Math.round(studentData.reduce((sum, s) => sum + parseInt(s.attendance), 0) / studentData.length) || 0
  };

  const StatCard = ({ title, value, subtitle, color, icon: Icon }) => (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-xl ${color.replace('text-', 'bg-').replace('-600', '-100')}`}>
          <Icon size={24} className={color} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="p-6 bg-gray-50 min-h-screen space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold mb-2">Student Reports & Analytics</h1>
            <p className="opacity-90">Comprehensive analysis of student performance and risk assessment</p>
          </div>
          <button
            onClick={generateReport}
            className="flex items-center px-4 py-2 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30 transition-colors"
          >
            <Download className="mr-2" size={16} />
            Export Report
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Students"
          value={stats.total}
          icon={Users}
          color="text-blue-600"
        />
        <StatCard
          title="High Risk"
          value={stats.highRisk}
          subtitle={`${Math.round((stats.highRisk/stats.total)*100)}% of students`}
          icon={TrendingUp}
          color="text-red-600"
        />
        <StatCard
          title="Average Grade"
          value={`${stats.avgGrade}%`}
          subtitle="Across all subjects"
          icon={BarChart3}
          color="text-green-600"
        />
        <StatCard
          title="Average Attendance"
          value={`${stats.avgAttendance}%`}
          subtitle="This semester"
          icon={Calendar}
          color="text-purple-600"
        />
      </div>

      {/* Charts Section */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold text-gray-800 mb-6">Performance Analytics</h2>
        <Charts studentData={studentData} />
      </div>

      {/* Filters and Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200 bg-gray-50">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <h3 className="text-lg font-semibold text-gray-800">Detailed Student Report</h3>
            
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="flex items-center space-x-2">
                <Filter size={16} className="text-gray-500" />
                <select
                  value={filterRisk}
                  onChange={(e) => setFilterRisk(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="All">All Risk Levels</option>
                  <option value="High">High Risk</option>
                  <option value="Medium">Medium Risk</option>
                  <option value="Low">Low Risk</option>
                </select>
              </div>
              
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="name">Sort by Name</option>
                <option value="average">Sort by Average</option>
                <option value="attendance">Sort by Attendance</option>
              </select>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Student</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Math</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Science</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">English</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Average</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Attendance</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk Level</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {sortedData.map((student, index) => {
                const avg = Math.round((parseInt(student.math) + parseInt(student.science) + parseInt(student.english)) / 3);
                return (
                  <tr key={index} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-8 h-8 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full flex items-center justify-center mr-3">
                          <span className="text-white text-sm font-semibold">{student.name.charAt(0)}</span>
                        </div>
                        <span className="font-medium text-gray-900">{student.name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-700 font-medium">{student.math}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-700 font-medium">{student.science}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-700 font-medium">{student.english}%</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`font-semibold ${
                        avg >= 80 ? 'text-green-600' : avg >= 60 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {avg}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-700 font-medium">{student.attendance}%</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-3 py-1 text-xs font-semibold rounded-full ${
                        student.risk === 'High' ? 'bg-red-100 text-red-800' :
                        student.risk === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {student.risk}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Reports;