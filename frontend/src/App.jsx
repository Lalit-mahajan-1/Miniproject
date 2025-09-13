import React, { useState } from 'react'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import UploadCSV from './pages/UploadCSV'
import CustomInput from './pages/CustomInput'
import Reports from './pages/Reports'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [studentData, setStudentData] = useState([])

  // Sample data for initial load
  const sampleData = [
    { name: 'Alice Johnson', math: 85, science: 78, english: 92, attendance: 95, risk: 'Low' },
    { name: 'Bob Smith', math: 62, science: 58, english: 65, attendance: 72, risk: 'High' },
    { name: 'Carol Davis', math: 78, science: 82, english: 75, attendance: 88, risk: 'Medium' },
    { name: 'David Wilson', math: 91, science: 89, english: 87, attendance: 98, risk: 'Low' },
    { name: 'Eva Brown', math: 45, science: 52, english: 48, attendance: 65, risk: 'High' },
    { name: 'Frank Miller', math: 73, science: 69, english: 71, attendance: 85, risk: 'Medium' },
    { name: 'Grace Lee', math: 88, science: 85, english: 90, attendance: 92, risk: 'Low' },
    { name: 'Henry Taylor', math: 56, science: 61, english: 59, attendance: 70, risk: 'High' }
  ]

  React.useEffect(() => {
    if (studentData.length === 0) {
      setStudentData(sampleData)
    }
  }, [])

  const calculateRisk = (student) => {
    const avgGrade = (parseInt(student.math) + parseInt(student.science) + parseInt(student.english)) / 3
    const attendance = parseInt(student.attendance)
    
    if (avgGrade < 60 || attendance < 75) return 'High'
    if (avgGrade < 75 || attendance < 85) return 'Medium'
    return 'Low'
  }

  const addStudent = (student) => {
    const newStudent = {
      ...student,
      risk: calculateRisk(student)
    }
    setStudentData(prev => [...prev, newStudent])
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard studentData={studentData} />
      case 'upload':
        return <UploadCSV studentData={studentData} setStudentData={setStudentData} calculateRisk={calculateRisk} />
      case 'custom':
        return <CustomInput addStudent={addStudent} />
      case 'reports':
        return <Reports studentData={studentData} />
      default:
        return <Dashboard studentData={studentData} />
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar setSidebarOpen={setSidebarOpen} />
        
        <main className="flex-1 overflow-y-auto bg-gray-50">
          {renderContent()}
        </main>
      </div>
    </div>
  )
}

export default App