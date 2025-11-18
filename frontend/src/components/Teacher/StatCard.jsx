import React from "react";
import { Card, Typography } from "@mui/material";

const StatCard = ({ title, value, color }) => (
  <Card
    style={{
      background: "#fff",
      borderTop: `5px solid ${color}`,
      boxShadow: "0 6px 15px rgba(0,0,0,0.1)",
      borderRadius: "10px",
      padding: "18px",
      textAlign: "center",
      transition: "transform 0.2s",
    }}
    onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.05)")}
    onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1.0)")}
  >
    <Typography variant="subtitle1" style={{ fontWeight: 600, color: "#444" }}>
      {title}
    </Typography>
    <Typography variant="h6" style={{ color, fontWeight: 700 }}>
      {value}
    </Typography>
  </Card>
);

export default StatCard;
