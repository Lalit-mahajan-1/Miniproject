const mongoose = require("mongoose");

const BatchSchema = new mongoose.Schema(
  {
    totalStudents: { type: Number, required: true },
    avg: { type: Number, required: true },
    high: { type: Number, required: true },
    med: { type: Number, required: true },
    low: { type: Number, required: true },
    highPct: { type: Number, required: true },
    medPct: { type: Number, required: true },
    lowPct: { type: Number, required: true },

    fileName: { type: String },
    createdBy: { type: String }
  },
  { timestamps: true }
);

module.exports = mongoose.model("Batch", BatchSchema);