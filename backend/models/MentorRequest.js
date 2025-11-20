// backend/models/MentorRequest.js
const mongoose = require("mongoose");
const { Schema } = mongoose;

const mentorRequestSchema = new Schema(
    {
        student: {
            type: Schema.Types.ObjectId,
            ref: "User",
            required: true,
        },

        status: {
            type: String,
            enum: ["pending", "approved", "rejected"],
            default: "pending",
        },

        // NEW FIELD: Assigned mentor (teacher)
        assignedTeacher: {
            type: Schema.Types.ObjectId,
            ref: "User",
            default: null,
        },
    },
    { timestamps: true }
);

module.exports = mongoose.model("MentorRequest", mentorRequestSchema);
