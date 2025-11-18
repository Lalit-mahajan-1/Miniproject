const mongoose = require("mongoose");

const StudentRiskSchema = new mongoose.Schema(
  {
    batch: { type: mongoose.Schema.Types.ObjectId, ref: "Batch", index: true },

    student_email: { type: String, index: true, required: true },
    student_id: { type: String },
    student_name: { type: String },

    predicted_risk_percentage: { type: Number, required: true },
    risk_category: { type: String, enum: ["High", "Medium", "Low"], required: true },

    attendance_percentage: { type: Number },
    weekly_self_study_hours: { type: Number },
    class_participation: { type: String },
    grade: { type: String },
    predicted_score: { type: Number },

    raw: { type: Object }
  },
  { timestamps: true }
);

StudentRiskSchema.index({ student_email: 1, createdAt: -1 });

module.exports = mongoose.model("StudentRisk", StudentRiskSchema);