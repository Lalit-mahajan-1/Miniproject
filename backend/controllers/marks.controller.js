// backend/src/controllers/marks.controller.js

const { nanoid } = require("nanoid");
const Mark = require("../models/Mark");

// This function defines the "key" for a mark record.
const getKeyForUpsert = (doc) => ({
  studentEmail: doc.studentEmail,
  subject: doc.subject,
  exam: doc.exam,
  academicYear: doc.academicYear,
});

exports.saveBatch = async (req, res) => {
  const { marks = [], fileName } = req.body;

  if (!Array.isArray(marks) || marks.length === 0) {
    return res.status(400).json({ message: "Marks data is missing or empty." });
  }

  const batchId = nanoid(10);

  // Prepare operations for bulk writing to the database
  const ops = marks.map((mark) => {
    return {
      updateOne: {
        filter: getKeyForUpsert(mark),
        update: {
          // Fields to update every time (if the record is found)
          $set: {
            studentName: mark.studentName,
            marksObtained: mark.marksObtained,
            outOf: mark.outOf,
            percentage: mark.percentage,
            batchId: batchId, // Update batchId to track the latest upload
            fileName: fileName,
          },
          // Fields to set ONLY on the initial creation of the record
          $setOnInsert: {
            studentEmail: mark.studentEmail,
            subject: mark.subject,
            exam: mark.exam,
            academicYear: mark.academicYear,
          },
        },
        upsert: true, // Update if exists, insert if not
      },
    };
  });

  try {
    const result = await Mark.bulkWrite(ops, { ordered: false });

    res.status(201).json({
      message: "Batch processed successfully.",
      batchId,
      inserted: result.upsertedCount,
      modified: result.modifiedCount,
    });
  } catch (error) {
    console.error("Error during marks bulk save:", error);
    // The specific error code is 40, but we can catch the generic name too
    if (error.name === 'MongoBulkWriteError' && error.code === 40) {
       return res.status(409).json({ message: "A data conflict occurred during the save operation. This is likely a server-side code issue." });
    }
    if (error.code === 11000) {
      return res.status(409).json({ message: "Duplicate data conflict. Some records might already exist." });
    }
    res.status(500).json({ message: "An error occurred while saving the marks." });
  }
};


// GET request to fetch marks (example) - This function remains the same
exports.getMarks = async (req, res) => {
  const { exam, academicYear } = req.query;

  // We need both exam and academicYear to perform a useful search
  if (!exam || !academicYear) {
    return res.status(400).json({ message: "Please provide both 'exam' and 'academicYear' to fetch records." });
  }

  try {
    const studentReports = await Mark.aggregate([
      // 1. Filter documents to only what we need (very important for performance)
      {
        $match: {
          exam: exam,
          academicYear: academicYear,
        },
      },
      // 2. Sort by subject for consistent order within the card
      {
        $sort: {
          subject: 1,
        },
      },
      // 3. Group by student to create one document per student
      {
        $group: {
          _id: "$studentEmail", // The unique key for each student
          studentName: { $first: "$studentName" }, // Get the student's name
          subjects: {
            // Create an array of this student's subjects and marks
            $push: {
              subject: "$subject",
              marksObtained: "$marksObtained",
              outOf: "$outOf",
              percentage: "$percentage",
            },
          },
        },
      },
      // 4. Sort the final student cards alphabetically
      {
        $sort: {
          studentName: 1,
        },
      },
    ]);

    res.status(200).json(studentReports);
  } catch (error) {
    console.error("Error fetching and aggregating marks:", error);
    res.status(500).json({ message: "Failed to fetch student reports." });
  }
};

// Add this new function to marks.controller.js
exports.getMyLatestMarks = async (req, res) => {
  const { email } = req.query;
  if (!email) {
    return res.status(400).json({ message: "Student email is required." });
  }

  try {
    // Step 1: Find the most recent record for this student to identify the latest exam
    const latestRecord = await Mark.findOne({ studentEmail: email.toLowerCase() }).sort({ createdAt: -1 });

    if (!latestRecord) {
      return res.json([]); // Return empty array if no marks found for the student
    }

    const { exam, academicYear } = latestRecord;

    // Step 2: Fetch all marks for that student for that latest exam and year
    const latestMarks = await Mark.find({
      studentEmail: email.toLowerCase(),
      exam,
      academicYear,
    }).select('subject marksObtained outOf percentage -_id'); // Select only the fields we need

    res.status(200).json(latestMarks);
  } catch (error) {
    console.error("Error fetching student's latest marks:", error);
    res.status(500).json({ message: "Failed to fetch student marks." });
  }
};