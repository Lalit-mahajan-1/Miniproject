// frontend/src/Components/Navbar.jsx
import React, { useContext, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";
import "./Navbar.css";
import {
  Drawer,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Divider,
  Avatar,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import CloseIcon from "@mui/icons-material/Close";
import HomeIcon from "@mui/icons-material/Home";
import FolderIcon from "@mui/icons-material/Folder";
import LogoutIcon from "@mui/icons-material/Logout";
import DescriptionIcon from "@mui/icons-material/Description"; // NEW

const Navbar = () => {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const toggleDrawer = (open) => (event) => {
    if (event?.type === "keydown" && (event.key === "Tab" || event.key === "Shift")) return;
    setDrawerOpen(open);
  };

  const handleNavigation = (path) => {
    navigate(path);
    setDrawerOpen(false);
  };

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <>
      <nav className="navbar" role="navigation" aria-label="main navigation">
        <div className="navbar-container">
          <div className="navbar-left">
            {user?.role === "teacher" && (
              <IconButton
                onClick={toggleDrawer(true)}
                className="menu-icon"
                aria-label="Open menu"
                size="large"
                edge="start"
              >
                <MenuIcon />
              </IconButton>
            )}
            <div className="navbar-brand" onClick={() => navigate("/")}>
              <h2>Student Risk System</h2>
            </div>
          </div>

          <div className="navbar-center">
            <p>Empowering Teachers with Predictive Insights</p>
          </div>

          <div className="navbar-right">
            <div className="user-section">
              <Avatar sx={{ bgcolor: "#4f9efc", width: 36, height: 36 }}>
                {user?.name?.[0]?.toUpperCase() || "U"}
              </Avatar>
              <div className="user-info">
                <span className="user-name">{user?.name}</span>
                <span className="user-role">{user?.role}</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={toggleDrawer(false)}
        transitionDuration={400}
        classes={{ paper: "glass-drawer" }}
      >
        <div className="drawer-header">
          <h3>Teacher Menu</h3>
          <IconButton onClick={toggleDrawer(false)}>
            <CloseIcon sx={{ color: "#fff" }} />
          </IconButton>
        </div>

        <Divider sx={{ bgcolor: "rgba(255,255,255,0.2)" }} />

        <List>
          <ListItem button onClick={() => handleNavigation("/teacher/home")}>
            <HomeIcon sx={{ mr: 2, color: "#4f9efc" }} />
            <ListItemText primary="Home" />
          </ListItem>

          <ListItem button onClick={() => handleNavigation("/teacher/records")}>
            <FolderIcon sx={{ mr: 2, color: "#4f9efc" }} />
            <ListItemText primary="Records" />
          </ListItem>

          {/* NEW */}
          <ListItem button onClick={() => handleNavigation("/teacher/syllabus")}>
            <DescriptionIcon sx={{ mr: 2, color: "#4f9efc" }} />
            <ListItemText primary="Add Syllabus" />
          </ListItem>
         <ListItem button onClick={() => handleNavigation("/teacher/marks-upload")}>
            <DescriptionIcon sx={{ mr: 2, color: "#4f9efc" }} />
            <ListItemText primary="Add Marks" />
          </ListItem>

          <Divider sx={{ bgcolor: "rgba(255,255,255,0.2)", my: 1 }} />

          <ListItem button onClick={handleLogout}>
            <LogoutIcon sx={{ mr: 2, color: "#ff5c5c" }} />
            <ListItemText primary="Logout" />
          </ListItem>
        </List>
      </Drawer>
    </>
  );
};

export default Navbar;