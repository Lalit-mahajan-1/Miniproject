import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";

const ResultsTable = ({ data, compact }) => {
  if (!data || data.length === 0) return null;

  const headers = Object.keys(data[0]);

  return (
    <TableContainer
      component={Paper}
      style={{
        boxShadow: "0 6px 12px rgba(0,0,0,0.05)",
        borderRadius: "10px",
      }}
    >
      <Table size={compact ? "small" : "medium"}>
        <TableHead>
          <TableRow>
            {headers.map((h) => (
              <TableCell
                key={h}
                style={{ fontWeight: 700, background: "#f1f3f5" }}
              >
                {h}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, i) => (
            <TableRow key={i}>
              {headers.map((h) => (
                <TableCell key={h}>{row[h]}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ResultsTable;
