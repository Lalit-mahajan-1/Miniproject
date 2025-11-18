// frontend/src/Components/Teacher/TeacherLayout.jsx
import React from "react";
import { Outlet } from "react-router-dom";
import { Box } from "@mui/material";
import Navbar from "../Navbar";

const TeacherLayout = () => {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        bgcolor: "#f8fafc", // soft background
      }}
    >
      {/* Navbar */}
      <Navbar />

      {/* Main content */}
      <Box sx={{ flex: 1, p: 3 }}>
        <Outlet />
      </Box>
    </Box>
  );
};

export default TeacherLayout;
