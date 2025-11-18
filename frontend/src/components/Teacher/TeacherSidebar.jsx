// frontend/src/Components/Teacher/TeacherSidebar.jsx
import { NavLink, useLocation } from 'react-router-dom';
import { Drawer, List, ListItemButton, ListItemText, Toolbar, Typography } from '@mui/material';

const drawerWidth = 240;

const TeacherSidebar = () => {
  const location = useLocation();

  const LinkItem = ({ to, label }) => {
    const selected = location.pathname.startsWith(to);
    return (
      <NavLink to={to} style={{ textDecoration: 'none', color: 'inherit' }}>
        <ListItemButton
          selected={selected}
          sx={{
            '&.Mui-selected': { backgroundColor: 'rgba(255,255,255,0.10)', color: '#fff' },
            '&.Mui-selected:hover': { backgroundColor: 'rgba(255,255,255,0.14)' },
            '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' },
          }}
        >
          <ListItemText primary={label} />
        </ListItemButton>
      </NavLink>
    );
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: '#0f172a',
          color: '#e5e7eb',
        },
      }}
    >
      <Toolbar sx={{ px: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#fff' }}>
          Teacher
        </Typography>
      </Toolbar>
      <List>
        <LinkItem to="/teacher/home" label="Home" />
        <LinkItem to="/teacher/records" label="Records" />
      </List>
    </Drawer>
  );
};

export default TeacherSidebar;