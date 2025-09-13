import React from 'react';
import { Users, AlertTriangle, Calendar, Award, TrendingUp, TrendingDown } from 'lucide-react';
import Charts from '../components/Charts';

const Dashboard = ({ studentData }) => {
  const totalStudents = studentData.length;
  const highRiskCount = studentData.filter(s => s.risk === 'High').length;
  const avgAttendance = totalStudents > 0 ? Math.round(studentData.reduce((sum, s) => sum + parseInt(s.attendance), 0) / totalStudents) : 0;
  const avgGrade = totalStudents > 0 ? Math.round(studentData.reduce((sum, s) => {
    const avg = (parseInt(s.math) + parseInt(s.science) + parseInt(s.english)) / 3;
    return sum + avg;
  }, 0) / totalStudents) : 0;

  const StatCard = ({ title, value, icon: Icon, color, bgColor, change, trend }) => (
    <div className={`${bgColor} p-6 rounded-xl shadow-sm border border-opacity-20 transition-all hover:shadow-md`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium opacity-80">{title}</p>
          <p className={`text-3xl font-bold ${color}`}>{value}</p>
          {change && (
            <div className="flex items-center mt-2">
              {trend === 'up' ? (
                <TrendingUp size={16} className="text-green-500 mr-1" />
              ) : (
                <TrendingDown size={16} className="text-red-500 mr-1" />
              )}
              <span className={`text-xs font-medium ${trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                {change}
              </span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-xl bg-white bg-opacity-20`}>
          <Icon size={28} className={color} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="p-6 space-y-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <h1 className="text-2xl font-bold mb-2">Welcome back, Teacher!</h1>
        <p className="opacity-90">Here's an overview of your students' performance and progress.</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Students"
          value={totalStudents}
          icon={Users}
          color="text-blue-700"
          bgColor="bg-gradient-to-br from-blue-50 to-blue-100"
          change="+2 this week"
          trend="up"
        />
        <StatCard
          title="At Risk Students"
          value={highRiskCount}
          icon={AlertTriangle}
          color="text-red-700"
          bgColor="bg-gradient-to-br from-red-50 to-red-100"
          change={`${Math.round((highRiskCount/totalStudents)*100)}% of total`}
        />
        <StatCard
          title="Avg Attendance"
          value={`${avgAttendance}%`}
          icon={Calendar}
          color="text-green-700"
          bgColor="bg-gradient-to-br from-green-50 to-green-100"
          change="+5% this month"
          trend="up"
        />
        <StatCard
          title="Avg Grade"
          value={`${avgGrade}%`}
          icon={Award}
          color="text-purple-700"
          bgColor="bg-gradient-to-br from-purple-50 to-purple-100"
          change="-2% this month"
          trend="down"
        />
      </div>

      {/* Charts */}
      <Charts studentData={studentData} />

      {/* Student Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200 bg-gray-50">
          <h3 className="text-lg font-semibold text-gray-800">Student Performance Overview</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Student</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Math</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Science</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">English</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Attendance</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk Level</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {studentData.slice(0, 10).map((student, index) => (
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
                  <td className="px-6 py-4 whitespace-nowrap text-gray-700 font-medium">{student.attendance}%</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-3 py-1 text-xs font-semibold rounded-full ${
                      student.risk === 'High' ? 'bg-red-100 text-red-800' :
                      student.risk === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {student.risk} Risk
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;