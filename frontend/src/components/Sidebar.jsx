// frontend/src/Components/Sidebar.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import {
  FiX,
  FiHome,
  FiFolder,
  FiLogOut,
  FiBookOpen,
  FiUsers
} from "react-icons/fi";
import { HiOutlineDocumentAdd } from "react-icons/hi";

import { AuthContext } from "../context/AuthContext";
import { useContext } from "react";

const Sidebar = ({ isDesktop, drawerOpen, setDrawerOpen }) => {
  const navigate = useNavigate();
  const { logout } = useContext(AuthContext);

  const handleNavigation = (path) => {
    navigate(path);
    if (!isDesktop) setDrawerOpen(false);
  };

  return (
    <>
      {/* Background overlay for mobile */}
      {!isDesktop && drawerOpen && (
        <div
          onClick={() => setDrawerOpen(false)}
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 animate-fade-in"
        />
      )}

      {/* Sidebar */}
      <div
        className={`
        fixed top-0 left-0 h-full w-72 bg-white shadow-xl border-r border-gray-200 z-50
        transform transition-transform duration-300
        ${drawerOpen ? "translate-x-0" : "-translate-x-full"}
      `}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800">Teacher Menu</h3>

          {!isDesktop && (
            <button
              onClick={() => setDrawerOpen(false)}
              className="p-2 rounded-md hover:bg-gray-100 transition"
            >
              <FiX className="w-6 h-6 text-gray-700" />
            </button>
          )}
        </div>

        {/* Menu list */}
        <div className="py-4">
          <button
            onClick={() => handleNavigation("/teacher/home")}
            className="menu-btn"
          >
            <FiHome className="icon-blue" />
            Home
          </button>

          <button
            onClick={() => handleNavigation("/teacher/records")}
            className="menu-btn"
          >
            <FiFolder className="icon-blue" />
            Records
          </button>

          <button
            onClick={() => handleNavigation("/teacher/syllabus")}
            className="menu-btn"
          >
            <HiOutlineDocumentAdd className="icon-blue" />
            Add Syllabus
          </button>

          <button
            onClick={() => handleNavigation("/teacher/marks-upload")}
            className="menu-btn"
          >
            <FiBookOpen className="icon-blue" />
            Add Marks
          </button>

          <button
  onClick={() => handleNavigation("/teacher/mentorship")}
  className="menu-btn"
>
  <FiUsers className="icon-blue" />
  Mentorship
</button>


          <div className="my-3 border-t border-gray-200" />

          <button
            onClick={() => {
              logout();
              navigate("/login");
            }}
            className="menu-btn text-red-600 font-medium"
          >
            <FiLogOut className="w-5 h-5" />
            Logout
          </button>
        </div>
      </div>

      {/* Small helper classes */}
      <style>{`
        .menu-btn {
          width: 100%;
          display: flex;
          gap: 12px;
          padding: 12px 20px;
          align-items: center;
          text-align: left;
          color: #374151;
          transition: 0.2s;
        }
        .menu-btn:hover {
          background: #f3f4f6;
        }
        .icon-blue {
          width: 20px;
          height: 20px;
          color: #3b82f6;
        }
      `}</style>
    </>
  );
};

export default Sidebar;
