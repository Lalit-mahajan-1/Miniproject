const mongoose = require("mongoose");

const MarkSchema = new mongoose.Schema(
  {
    // Student Identifier
    studentEmail: { type: String, required: true, index: true, lowercase: true, trim: true },
    studentName: { type: String },

    // Mark Details
    subject: { type: String, required: true, index: true, uppercase: true },
    marksObtained: { type: Number, required: true },
    outOf: { type: Number, required: true },
    percentage: { type: Number, required: true },

    // Context for the mark
    exam: { type: String, required: true, index: true },
    academicYear: { type: String, required: true, index: true },

    // Upload Metadata
    batchId: { type: String, required: true, index: true },
    fileName: { type: String },
    uploadedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }, // Optional: Link to user
  },
  { timestamps: true }
);

// This compound unique index is CRITICAL. It prevents a teacher from accidentally
// uploading the same mark for the same student, subject, exam, and year twice.
MarkSchema.index(
  { studentEmail: 1, subject: 1, exam: 1, academicYear: 1 },
  { unique: true }
);

MarkSchema.set("toJSON", {
  transform: (_, ret) => {
    ret.id = ret._id;
    delete ret._id;
    delete ret.__v;
    return ret;
  },
});

module.exports = mongoose.model("Mark", MarkSchema);