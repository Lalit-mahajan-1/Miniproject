import React from 'react';
import { Menu, BookOpen, Bell, Search } from 'lucide-react';

const Navbar = ({ setSidebarOpen }) => {
  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="flex items-center justify-between h-16 px-6">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-colors"
          >
            <Menu size={20} />
          </button>
          
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <BookOpen className="text-white" size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800">EduAnalytics</h1>
              <p className="text-xs text-gray-500">Teacher Dashboard</p>
            </div>
          </div>
        </div>

        {/* Center - Search */}
        <div className="hidden md:flex flex-1 max-w-md mx-8">
          <div className="relative w-full">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
            <input
              type="text"
              placeholder="Search students..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          <button className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors relative">
            <Bell size={20} />
            <span className="absolute -top-1 -right-1 h-4 w-4 bg-red-500 text-white rounded-full text-xs flex items-center justify-center">
              3
            </span>
          </button>
          
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center shadow-md">
              <span className="text-white font-semibold text-sm">T</span>
            </div>
            <div className="hidden sm:block">
              <p className="text-sm font-medium text-gray-700">Teacher</p>
              <p className="text-xs text-gray-500">Online</p>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;