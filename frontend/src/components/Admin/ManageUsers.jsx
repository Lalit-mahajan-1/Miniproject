// frontend/src/Components/Admin/ManageUsers.jsx
import { useState, useEffect } from "react";
import axios from "axios";
import Navbar from "../Navbar";
import { FiUsers, FiUserCheck, FiUserX, FiUserPlus, FiSearch, FiTrash2, FiMenu, FiX, FiUser, FiStar } from "react-icons/fi";

const ManageUsers = () => {
    const [users, setUsers] = useState([]);
    const [showForm, setShowForm] = useState(false);
    const [search, setSearch] = useState("");
    const [formData, setFormData] = useState({
        name: "",
        email: "",
        password: "",
        role: "student",
        studentId: "",
    });
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState("");
    const [activeSection, setActiveSection] = useState("users"); // 'users' | 'mentors'
    const [sidebarOpen, setSidebarOpen] = useState(false);

    // Mentor-related state
    const [mentorRequests, setMentorRequests] = useState([]);
    const [mentorsMap, setMentorsMap] = useState([]);
    const [selectedRequest, setSelectedRequest] = useState(null);
    const [selectedTeacherId, setSelectedTeacherId] = useState("");
    const [assignLoading, setAssignLoading] = useState(false);

    const API_BASE = import.meta.env.VITE_API_URL;

    useEffect(() => {
        fetchUsers();
    }, []);

    useEffect(() => {
        if (activeSection === "mentors") {
            fetchMentorData();
        }
    }, [activeSection]);

    const fetchUsers = async () => {
        try {
            const res = await axios.get(`${API_BASE}/api/auth/users`);
            setUsers(res.data);
        } catch (error) {
            console.error("Error fetching users:", error);
        }
    };

    const fetchMentorData = async () => {
        try {
            const [reqRes, mapRes] = await Promise.all([
                axios.get(`${API_BASE}/api/admin/mentor-requests`),
                axios.get(`${API_BASE}/api/admin/mentors-map`),
            ]);
            setMentorRequests(reqRes.data);
            setMentorsMap(mapRes.data);
        } catch (error) {
            console.error("Error fetching mentor data:", error);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setMessage("");

        try {
            await axios.post(`${API_BASE}/api/auth/register`, formData);
            setMessage("User created successfully!");
            setFormData({
                name: "",
                email: "",
                password: "",
                role: "student",
                studentId: "",
            });
            setShowForm(false);
            fetchUsers();
        } catch (error) {
            setMessage(error.response?.data?.message || "Error creating user");
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id) => {
        if (!confirm("Are you sure you want to delete this user?")) return;
        try {
            await axios.delete(`${API_BASE}/api/auth/users/${id}`);
            setMessage("User deleted successfully");
            fetchUsers();
        } catch (error) {
            setMessage("Error deleting user");
        }
    };

    const handleAssignMentor = async (e) => {
        e.preventDefault();
        if (!selectedRequest || !selectedTeacherId) return;

        setAssignLoading(true);
        setMessage("");

        try {
            await axios.post(`${API_BASE}/api/admin/assign-mentor`, {
                studentId: selectedRequest.student._id,
                teacherId: selectedTeacherId,
                requestId: selectedRequest._id,
            });

            setMessage("Mentor assigned successfully");
            setSelectedRequest(null);
            setSelectedTeacherId("");
            await fetchMentorData();
            await fetchUsers();
        } catch (error) {
            setMessage(error.response?.data?.message || "Error assigning mentor");
        } finally {
            setAssignLoading(false);
        }
    };

    const filteredUsers = users.filter(
        (u) => u.name.toLowerCase().includes(search.toLowerCase()) || u.email.toLowerCase().includes(search.toLowerCase())
    );

    const count = {
        students: users.filter((u) => u.role === "student").length,
        teachers: users.filter((u) => u.role === "teacher").length,
        admins: users.filter((u) => u.role === "admin").length,
    };

    const teacherUsers = users.filter((u) => u.role === "teacher");

    useEffect(() => {
        if (message) {
            const timer = setTimeout(() => {
                setMessage("");
            }, 3000);
            return () => clearTimeout(timer);
        }
    }, [message]);

    return (
        <>
            <Navbar />

            <div className="flex min-h-[calc(100vh-100px)] bg-slate-50 relative">
                {/* Mobile overlay */}
                {sidebarOpen && (
                    <div
                        className="fixed inset-0 bg-black/40 z-20 lg:hidden"
                        onClick={() => setSidebarOpen(false)}
                    />
                )}

                {/* Slide-out Sidebar */}
                <aside
                    className={`fixed z-30 inset-y-0 left-0 w-64 bg-white border-r border-gray-200 transform transition-transform duration-200 lg:static lg:translate-x-0 ${
                        sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
                    }`}
                >
                    <div className="h-full flex flex-col">
                        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
                            <span className="font-semibold text-gray-800">Admin Panel</span>
                            <button
                                className="lg:hidden p-1 rounded-md hover:bg-gray-100"
                                onClick={() => setSidebarOpen(false)}
                            >
                                <FiX />
                            </button>
                        </div>

                        <nav className="flex-1 px-3 py-4 space-y-2">
                            <button
                                onClick={() => {
                                    setActiveSection("users");
                                    setSidebarOpen(false);
                                }}
                                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium ${
                                    activeSection === "users" ? "bg-indigo-50 text-indigo-600" : "text-gray-600 hover:bg-gray-50"
                                }`}
                            >
                                <FiUsers />
                                Users
                            </button>

                            <button
                                onClick={() => {
                                    setActiveSection("mentors");
                                    setSidebarOpen(false);
                                }}
                                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium ${
                                    activeSection === "mentors" ? "bg-indigo-50 text-indigo-600" : "text-gray-600 hover:bg-gray-50"
                                }`}
                            >
                                <FiUser />
                                Mentors
                            </button>
                        </nav>
                    </div>
                </aside>

                {/* Main content */}
                <main className="flex-1 max-w-7xl mx-auto w-full px-4 lg:px-8 py-6 lg:py-10">
                    {/* Top bar inside main */}
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                            <button
                                className="lg:hidden p-2 rounded-lg border border-gray-200 bg-white shadow-sm"
                                onClick={() => setSidebarOpen(true)}
                            >
                                <FiMenu />
                            </button>
                            <div>
                                <h1 className="text-2xl lg:text-3xl font-bold text-gray-900">
                                    {activeSection === "users" ? "User Management" : "Mentor Management"}
                                </h1>
                                <p className="text-gray-500 text-sm">
                                    {activeSection === "users" ? "Manage roles, permissions & access" : "Handle mentor requests and assignments"}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Counter cards (always visible) */}
                    {activeSection === "users" ? (
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-8">
                            <div className="bg-white rounded-xl shadow p-6 flex items-center gap-4 border border-gray-100">
                                <FiUsers
                                    className="text-indigo-600"
                                    size={32}
                                />
                                <div>
                                    <p className="text-gray-500 text-sm">Total Students</p>
                                    <h2 className="text-2xl font-semibold">{count.students}</h2>
                                </div>
                            </div>

                            <div className="bg-white rounded-xl shadow p-6 flex items-center gap-4 border border-gray-100">
                                <FiUserCheck
                                    className="text-blue-600"
                                    size={32}
                                />
                                <div>
                                    <p className="text-gray-500 text-sm">Total Teachers</p>
                                    <h2 className="text-2xl font-semibold">{count.teachers}</h2>
                                </div>
                            </div>

                            <div className="bg-white rounded-xl shadow p-6 flex items-center gap-4 border border-gray-100">
                                <FiUserX
                                    className="text-rose-600"
                                    size={32}
                                />
                                <div>
                                    <p className="text-gray-500 text-sm">Total Admins</p>
                                    <h2 className="text-2xl font-semibold">{count.admins}</h2>
                                </div>
                            </div>
                        </div>
                    ) : null}

                    {message && <div className="mb-4 p-3 bg-indigo-100 text-indigo-700 rounded-lg text-sm">{message}</div>}

                    {/* SECTION: USERS */}
                    {activeSection === "users" && (
                        <>
                            {/* Search + Add button */}
                            <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between mb-6 gap-4">
                                <div className="relative w-full sm:w-72">
                                    <FiSearch className="absolute left-3 top-3 text-gray-400" />
                                    <input
                                        type="text"
                                        placeholder="Search users..."
                                        value={search}
                                        onChange={(e) => setSearch(e.target.value)}
                                        className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white"
                                    />
                                </div>

                                <button
                                    onClick={() => setShowForm(!showForm)}
                                    className="flex items-center justify-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition shadow-sm"
                                >
                                    <FiUserPlus /> {showForm ? "Cancel" : "Add User"}
                                </button>
                            </div>

                            {/* Form Card */}
                            {showForm && (
                                <div className="bg-white border border-gray-100 shadow rounded-xl p-6 mb-6">
                                    <h2 className="text-xl font-semibold mb-4">Create New User</h2>

                                    <form
                                        onSubmit={handleSubmit}
                                        className="grid grid-cols-1 sm:grid-cols-2 gap-4"
                                    >
                                        <input
                                            type="text"
                                            placeholder="Full Name"
                                            className="border p-3 rounded-lg focus:ring-indigo-500 focus:ring-2"
                                            value={formData.name}
                                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                            required
                                        />

                                        <input
                                            type="email"
                                            placeholder="Email"
                                            className="border p-3 rounded-lg focus:ring-indigo-500 focus:ring-2"
                                            value={formData.email}
                                            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                            required
                                        />

                                        <input
                                            type="password"
                                            placeholder="Password"
                                            className="border p-3 rounded-lg focus:ring-indigo-500 focus:ring-2"
                                            value={formData.password}
                                            onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                                            required
                                        />

                                        <select
                                            className="border p-3 rounded-lg focus:ring-indigo-500 focus:ring-2"
                                            value={formData.role}
                                            onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                                        >
                                            <option value="student">Student</option>
                                            <option value="teacher">Teacher</option>
                                            <option value="admin">Admin</option>
                                        </select>

                                        {formData.role === "student" && (
                                            <input
                                                type="text"
                                                placeholder="Student ID"
                                                className="border p-3 rounded-lg focus:ring-indigo-500 focus:ring-2"
                                                value={formData.studentId}
                                                onChange={(e) =>
                                                    setFormData({
                                                        ...formData,
                                                        studentId: e.target.value,
                                                    })
                                                }
                                            />
                                        )}

                                        <button
                                            type="submit"
                                            className="col-span-full bg-indigo-600 text-white py-3 rounded-lg hover:bg-indigo-700 transition"
                                            disabled={loading}
                                        >
                                            {loading ? "Creating..." : "Create User"}
                                        </button>
                                    </form>
                                </div>
                            )}

                            {/* User Table */}
                            <div className="bg-white border border-gray-100 shadow rounded-xl overflow-hidden">
                                <table className="w-full">
                                    <thead className="bg-indigo-600 text-white">
                                        <tr>
                                            <th className="py-3 px-4 text-left text-sm font-medium">Name</th>
                                            <th className="py-3 px-4 text-left text-sm font-medium">Email</th>
                                            <th className="py-3 px-4 text-left text-sm font-medium">Role</th>
                                            <th className="py-3 px-4 text-left text-sm font-medium">Student ID</th>
                                            <th className="py-3 px-4 text-left text-sm font-medium">Actions</th>
                                        </tr>
                                    </thead>

                                    <tbody>
                                        {filteredUsers.map((user) => (
                                            <tr
                                                key={user._id}
                                                className="border-b last:border-0 hover:bg-gray-50"
                                            >
                                                <td className="py-3 px-4 text-sm">{user.name}</td>
                                                <td className="py-3 px-4 text-sm">{user.email}</td>
                                                <td className="py-3 px-4 text-sm">
                                                    <span
                                                        className={`px-3 py-1 rounded-full text-xs font-semibold capitalize ${
                                                            user.role === "admin"
                                                                ? "bg-red-100 text-red-700"
                                                                : user.role === "teacher"
                                                                ? "bg-blue-100 text-blue-700"
                                                                : "bg-green-100 text-green-700"
                                                        }`}
                                                    >
                                                        {user.role}
                                                    </span>
                                                </td>
                                                <td className="py-3 px-4 text-sm">{user.studentId || "-"}</td>
                                                <td className="py-3 px-4 text-sm">
                                                    <button
                                                        onClick={() => handleDelete(user._id)}
                                                        className="flex items-center gap-1 bg-rose-500 text-white px-3 py-1.5 rounded-lg hover:bg-rose-600 transition text-xs"
                                                    >
                                                        <FiTrash2 /> Delete
                                                    </button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </>
                    )}

                    {/* SECTION: MENTORS */}
                    {activeSection === "mentors" && (
                        <div className="grid gap-6 lg:grid-cols-2">
                            {/* Mentor Requests */}
                            <div className="bg-white border border-gray-100 shadow rounded-xl p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h2 className="text-lg font-semibold flex items-center gap-2">
                                        <FiStar className="text-yellow-500" /> Mentor Requests
                                    </h2>
                                </div>

                                {mentorRequests.length === 0 ? (
                                    <p className="text-sm text-gray-500">No pending mentor requests.</p>
                                ) : (
                                    <div className="space-y-3 max-h-80 overflow-auto pr-1">
                                        {mentorRequests.map((req) => (
                                            <div
                                                key={req._id}
                                                className={`border rounded-lg p-3 flex items-center justify-between ${
                                                    selectedRequest?._id === req._id ? "border-indigo-500 bg-indigo-50" : "border-gray-200"
                                                }`}
                                            >
                                                <div>
                                                    <p className="text-sm font-medium text-gray-900">{req.student.name}</p>
                                                    <p className="text-xs text-gray-500">
                                                        {req.student.email} • {req.student.studentId || "N/A"}
                                                    </p>
                                                </div>
                                                <button
                                                    onClick={() => setSelectedRequest(req)}
                                                    className="text-xs px-3 py-1.5 rounded-md border border-indigo-500 text-indigo-600 hover:bg-indigo-50"
                                                >
                                                    {selectedRequest?._id === req._id ? "Selected" : "Assign"}
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Assign mentor form */}
                                <div className="mt-5 border-t pt-4">
                                    <h3 className="text-sm font-semibold mb-2">Assign Mentor to Selected Student</h3>

                                    {selectedRequest ? (
                                        <>
                                            <p className="text-xs text-gray-500 mb-2">
                                                Student:{" "}
                                                <span className="font-medium">
                                                    {selectedRequest.student.name} ({selectedRequest.student.studentId || "N/A"})
                                                </span>
                                            </p>

                                            <form
                                                onSubmit={handleAssignMentor}
                                                className="space-y-3 text-sm"
                                            >
                                                <select
                                                    className="w-full border p-2.5 rounded-lg focus:ring-indigo-500 focus:ring-2"
                                                    value={selectedTeacherId}
                                                    onChange={(e) => setSelectedTeacherId(e.target.value)}
                                                    required
                                                >
                                                    <option value="">Select mentor (teacher)</option>
                                                    {teacherUsers.map((t) => (
                                                        <option
                                                            key={t._id}
                                                            value={t._id}
                                                        >
                                                            {t.name} • {t.email}
                                                        </option>
                                                    ))}
                                                </select>

                                                <button
                                                    type="submit"
                                                    disabled={assignLoading || !selectedTeacherId}
                                                    className="w-full bg-indigo-600 text-white py-2.5 rounded-lg hover:bg-indigo-700 transition disabled:opacity-60"
                                                >
                                                    {assignLoading ? "Assigning..." : "Assign Mentor"}
                                                </button>
                                            </form>
                                        </>
                                    ) : (
                                        <p className="text-xs text-gray-500">Select a request from the list above to assign a mentor.</p>
                                    )}
                                </div>
                            </div>

                            {/* Mentors & Students Map */}
                            <div className="bg-white border border-gray-100 shadow rounded-xl p-6">
                                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                    <FiUser className="text-indigo-600" /> Mentors & Students
                                </h2>

                                {mentorsMap.length === 0 ? (
                                    <p className="text-sm text-gray-500">No mentor-student mappings yet.</p>
                                ) : (
                                    <div className="space-y-4 max-h-96 overflow-auto pr-1">
                                        {mentorsMap.map((entry) => (
                                            <div
                                                key={entry.teacher._id}
                                                className="border border-gray-200 rounded-lg p-4"
                                            >
                                                <div className="flex items-center justify-between mb-2">
                                                    <div>
                                                        <p className="font-medium text-sm text-gray-900 flex items-center gap-2">
                                                            <FiUser className="text-indigo-500" />
                                                            {entry.teacher.name}
                                                        </p>
                                                        <p className="text-xs text-gray-500">{entry.teacher.email}</p>
                                                    </div>
                                                    <span className="text-xs px-2 py-1 rounded-full bg-indigo-50 text-indigo-600">
                                                        {entry.students.length} students
                                                    </span>
                                                </div>

                                                {/* {entry.students.length > 0 ? (
                                                    <ul className="mt-2 space-y-1">
                                                        {entry.students.map((s) => (
                                                            <li
                                                                key={s._id}
                                                                className="text-xs text-gray-700 flex justify-between"
                                                            >
                                                                <span>{s.name}</span>
                                                                <span className="text-gray-400">{s.studentId || ""}</span>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                ) : (
                                                    <p className="text-xs text-gray-400">No students assigned yet.</p>
                                                )} */}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </>
    );
};

export default ManageUsers;
