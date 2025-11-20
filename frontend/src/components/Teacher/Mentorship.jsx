import React, { useEffect, useState } from "react";
import axios from "axios";
import { FiTrash2, FiUsers } from "react-icons/fi";
import { useContext } from "react";
import { AuthContext } from "../../context/AuthContext";

const Mentorship = () => {
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
    const { user } = useContext(AuthContext);
  

  const teacherId = user?.id;  // OR pull from localStorage


  const fetchMentees = async () => {
    console.log("Fetching mentees for teacher ID:", user);
    try {
      const api = import.meta.env.VITE_API_URL;
      const res = await axios.get(
  `${api}/api/teacher/mentees?teacherId=${teacherId}`
);

      setStudents(res.data.data || []);
    } catch (err) {
      console.error("Error fetching mentees:", err);
    } finally {
      setLoading(false);
    }
  };

const removeStudent = async (reqId) => {
    if (!confirm("Remove this student from mentorship?")) return;

    try {
        const api = import.meta.env.VITE_API_URL;

        await axios.post(`${api}/api/teacher/remove`, {
            requestId: reqId
        });

        setStudents((prev) => prev.filter((st) => st._id !== reqId));
    } catch (err) {
        console.error("Error removing student:", err);
    }
};


  useEffect(() => {
    fetchMentees();
  }, []);

  return (
    <div className="p-6 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-slate-900">Mentorship</h1>

        <div className="flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-full text-sm">
          <FiUsers className="text-slate-600" />
          <span>{students.length} Assigned Students</span>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="text-center py-10 text-slate-500">Loading...</div>
      )}

      {/* Table */}
      {!loading && students.length === 0 && (
        <div className="text-center py-10 text-slate-500">
          No students assigned to you yet.
        </div>
      )}

      {students.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-600 uppercase text-xs">
              <tr>
                <th className="py-3 px-4 text-left">Name</th>
                <th className="py-3 px-4 text-left">Email</th>
                <th className="py-3 px-4 text-left">Department</th>
                <th className="py-3 px-4 text-left">Year</th>
                <th className="py-3 px-4 text-left">Phone</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>

            <tbody>
              {students.map((item) => (
                <tr
                  key={item._id}
                  className="border-t border-gray-100 hover:bg-slate-50 transition"
                >
                  <td className="px-4 py-3">
                    {item.student?.name || "Unknown"}
                  </td>
                  <td className="px-4 py-3">{item.student?.email}</td>
                  <td className="px-4 py-3">{item.student?.department}</td>
                  <td className="px-4 py-3">{item.student?.year}</td>
                  <td className="px-4 py-3">{item.student?.phone}</td>

                  <td className="px-4 py-3 text-right">
                    <button
  onClick={() => removeStudent(item._id)}
  className="p-2 text-red-600 hover:text-red-800"
>
  <FiTrash2 />
</button>

                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
          animation: fadeUp .35s cubic-bezier(.2,.9,.2,1) forwards;
        }
      `}</style>
    </div>
  );
};

export default Mentorship;
