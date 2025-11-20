// frontend/src/Layout/TeacherLayout.jsx
import React, { useContext, useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import Navbar from "../Navbar";
import Sidebar from "../Sidebar"; // We'll extract it from Navbar

const TeacherLayout = () => {
  const [isDesktop, setIsDesktop] = useState(window.innerWidth >= 1024);
  const [drawerOpen, setDrawerOpen] = useState(window.innerWidth >= 1024);

  useEffect(() => {
    const handleResize = () => {
      const desktop = window.innerWidth >= 1024;
      setIsDesktop(desktop);
      setDrawerOpen(desktop);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="w-full h-full">
      {/* Navbar */}
      <Navbar drawerOpen={drawerOpen} setDrawerOpen={setDrawerOpen} />

      {/* Sidebar */}
      <Sidebar isDesktop={isDesktop} drawerOpen={drawerOpen} setDrawerOpen={setDrawerOpen} />

      {/* PAGE CONTENT */}
      <main
        className={`
          p-6 transition-all 
          ${isDesktop ? "lg:pl-80" : ""}
        `}
      >
        <Outlet />
      </main>
    </div>
  );
};

export default TeacherLayout;
