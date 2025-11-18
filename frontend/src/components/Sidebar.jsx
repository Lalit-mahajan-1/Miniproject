import React from 'react';
import { X, Home, Upload, UserPlus, BarChart3, Settings, LogOut } from 'lucide-react';

const Sidebar = ({ activeTab, setActiveTab, sidebarOpen, setSidebarOpen }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home, color: 'text-blue-600' },
    { id: 'upload', label: 'Upload CSV', icon: Upload, color: 'text-green-600' },
    { id: 'custom', label: 'Add Student', icon: UserPlus, color: 'text-purple-600' },
    { id: 'reports', label: 'Reports', icon: BarChart3, color: 'text-orange-600' },
  ];

  const handleItemClick = (id) => {
    setActiveTab(id);
    setSidebarOpen(false);
  };

  return (
    <>
      {/* Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-gray-900 transform transition-transform duration-300 ease-in-out ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } lg:translate-x-0 lg:static lg:inset-0`}>
        
        {/* Header */}
        <div className="flex items-center justify-between h-16 px-6 bg-gray-800">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">E</span>
            </div>
            <span className="text-white font-semibold">EduAnalytics</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden text-gray-300 hover:text-white p-1"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="mt-8 px-4">
          <div className="space-y-2">
            {menuItems.map(({ id, label, icon: Icon, color }) => (
              <button
                key={id}
                onClick={() => handleItemClick(id)}
                className={`w-full flex items-center px-4 py-3 rounded-lg text-left transition-all duration-200 ${
                  activeTab === id 
                    ? 'bg-gray-800 text-white shadow-lg border-l-4 border-blue-500' 
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <Icon size={20} className={`mr-3 ${activeTab === id ? 'text-blue-400' : color}`} />
                <span className="font-medium">{label}</span>
                {activeTab === id && (
                  <div className="ml-auto w-2 h-2 bg-blue-400 rounded-full"></div>
                )}
              </button>
            ))}
          </div>

          {/* Bottom Section */}
          <div className="absolute bottom-8 left-4 right-4 space-y-2">
            <button className="w-full flex items-center px-4 py-3 text-gray-300 hover:bg-gray-800 hover:text-white rounded-lg transition-colors">
              <Settings size={20} className="mr-3 text-gray-400" />
              <span>Settings</span>
            </button>
            <button className="w-full flex items-center px-4 py-3 text-gray-300 hover:bg-red-600 hover:text-white rounded-lg transition-colors">
              <LogOut size={20} className="mr-3 text-red-400" />
              <span>Logout</span>
            </button>
          </div>
        </nav>
      </div>
    </>
  );
};

export default Sidebar;