const express = require("express");
const Batch = require("../models/Batch");
const StudentRisk = require("../models/StudentRisk");
const {
  toNumber,
  riskCategory,
  detectEmail,
  pickName,
  pickStudentId,
  pick
} = require("../utils/normalizers");

const router = express.Router();

// POST /api/student-risks/save-batch
router.post("/save-batch", async (req, res, next) => {
  try {
    const { students = [], fileName, createdBy } = req.body || {};
    if (!Array.isArray(students) || students.length === 0) {
      return res.status(400).json({ message: "students array is required." });
    }

    const normalized = [];
    for (const row of students) {
      const email = (detectEmail(row) || String(row?.email || "")).trim().toLowerCase();
      if (!email) continue;

      const risk = toNumber(row.predicted_risk_percentage) ?? 0;

      // Map your CSV headers into the dashboard fields
      const attendance = toNumber(
        pick(row, ["Attendance_Percentage", "attendance_percentage", "attendance"])
      );
      const selfStudy = toNumber(
        pick(row, ["Study_Hours_Per_Week", "weekly_self_study_hours", "self_study_hours"])
      );
      const participation = pick(row, ["Lecture_Participation", "class_participation", "participation"]);
      const grade = pick(row, ["grade", "Previous_Semester_Grades"]);
      const predictedScore = toNumber(
        pick(row, ["predicted_score", "Current_Internal_Score", "score", "Marks", "marks"])
      );

      const studentId = pickStudentId(row);
      const name = pickName(row) || row.Student_Name || undefined;

      normalized.push({
        student_email: email,
        student_id: studentId,
        student_name: name,
        predicted_risk_percentage: Number(risk || 0),
        risk_category: riskCategory(risk),
        attendance_percentage: attendance,
        weekly_self_study_hours: selfStudy,
        class_participation: participation ? String(participation) : undefined,
        grade: grade ? String(grade) : undefined,
        predicted_score: predictedScore,
        raw: row
      });
    }

    if (normalized.length === 0) {
      return res.status(400).json({ message: "No valid rows with email detected." });
    }

    // Stats
    const risks = normalized.map(r => Number(r.predicted_risk_percentage || 0));
    const total = risks.length;
    const avg = risks.reduce((a, b) => a + b, 0) / total;
    const high = risks.filter(v => v > 70).length;
    const med  = risks.filter(v => v > 40 && v <= 70).length;
    const low  = risks.filter(v => v <= 40).length;
    const highPct = (high / total) * 100;
    const medPct  = (med / total) * 100;
    const lowPct  = (low / total) * 100;

    const batch = await Batch.create({
      totalStudents: total,
      avg,
      high,
      med,
      low,
      highPct,
      medPct,
      lowPct,
      fileName,
      createdBy
    });

    const docs = normalized.map(n => ({ ...n, batch: batch._id }));
    await StudentRisk.insertMany(docs);

    return res.json({
      batchId: String(batch._id),
      summary: { totalStudents: total, avg, high, med, low, highPct, medPct, lowPct },
      count: docs.length
    });
  } catch (err) {
    next(err);
  }
});

// GET /api/student-risks/batch/:batchId
router.get("/batch/:batchId", async (req, res, next) => {
  try {
    const { batchId } = req.params;
    const batch = await Batch.findById(batchId).lean();
    if (!batch) return res.status(404).json({ message: "Batch not found" });

    const students = await StudentRisk.find({ batch: batchId })
      .sort({ createdAt: -1 })
      .lean();

    return res.json({ batch, students });
  } catch (err) {
    next(err);
  }
});

// GET /api/student-risks/batches
router.get("/batches", async (req, res, next) => {
  try {
    const batches = await Batch.find({})
      .sort({ createdAt: -1 })
      .limit(50)
      .lean();
    return res.json({ items: batches });
  } catch (err) {
    next(err);
  }
});

// GET /api/student-risks/me?email=<student_email>
router.get("/me", async (req, res, next) => {
  try {
    const email = (req.get("X-User-Email") || req.query.email || "").trim().toLowerCase();
    if (!email) {
      return res.status(400).json({ message: "Email is required via X-User-Email header or ?email query parameter." });
    }

    const history = await StudentRisk.find({ student_email: email })
      .sort({ createdAt: -1 })
      .limit(50)
      .lean();

    const latest = history[0] || null;

    // Aggregate summary
    let sum = 0, count = 0, high = 0, med = 0, low = 0;
    for (const h of history) {
      const r = Number(h.predicted_risk_percentage || 0);
      sum += r; count += 1;
      if (r > 70) high++;
      else if (r > 40) med++;
      else low++;
    }
    const avgRisk = count ? sum / count : 0;

    const latestOut = latest ? {
      ...latest,
      name: latest.student_name || latest.raw?.Student_Name || latest.raw?.name || latest.student_email
    } : null;

    return res.json({
      latest: latestOut,
      history,
      avgRisk,
      highRiskCount: high,
      mediumRiskCount: med,
      lowRiskCount: low
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;