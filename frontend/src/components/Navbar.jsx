// frontend/src/Components/Navbar.jsx
import React, { useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import { FiMenu } from "react-icons/fi";

const Navbar = ({ drawerOpen, setDrawerOpen }) => {
  const { user } = useContext(AuthContext);

  return (
    <nav className="w-full bg-white shadow-sm border-b border-gray-100 px-6 py-3 flex items-center justify-between sticky top-0 z-40 lg:pl-80">
      {/* Left Section */}
      <div className="flex items-center gap-4">
        {/* Hamburger only on mobile */}
        <button
          onClick={() => setDrawerOpen(!drawerOpen)}
          className="lg:hidden p-2 rounded-md hover:bg-gray-100 transition"
        >
          <FiMenu className="w-6 h-6 text-gray-700" />
        </button>

        <h2 className="text-xl font-semibold text-gray-800">Student Risk System</h2>
      </div>

      {/* Center */}
      <div className="hidden md:block text-gray-500 text-sm">
        Empowering Teachers with Predictive Insights
      </div>

      {/* Right = user */}
      <div className="flex items-center gap-3 bg-gray-50 px-3 py-1.5 rounded-xl border border-gray-200">
        <div className="w-9 h-9 flex items-center justify-center rounded-full bg-blue-500 text-white font-medium">
          {user?.name?.[0] || "U"}
        </div>

        <div className="flex flex-col leading-tight">
          <span className="text-sm font-medium text-gray-800">{user?.name}</span>
          <span className="text-xs text-gray-500">{user?.role}</span>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
