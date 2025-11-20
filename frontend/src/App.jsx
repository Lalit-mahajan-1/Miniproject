// frontend/src/pages/App.jsx
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Login from "./components/Auth/Login";
import PrivateRoute from "./Components/Auth/PrivateRoute";

// TEACHER
import TeacherLayout from "./components/Teacher/TeacherLayout";
import FileForm from "./components/Teacher/FileForm";
import Records from "./Components/Teacher/Records";
import AddSyllabus from "./Components/Teacher/AddSyllabus";
import MarksUpload from "./Components/Teacher/MarksUpload"; // <-- NEW: Import the component
import Mentorship from "./components/Teacher/Mentorship"; // <-- NEW: Import the Mentorship component

// OTHER roles
import StudentDashboard from "./Components/Student/Dashboard";
import ManageUsers from "./components/Admin/ManageUsers";

function App() {
    return (
        <AuthProvider>
            <Router>
                <Routes>
                    <Route
                        path="/login"
                        element={<Login />}
                    />

                    <Route
    path="/teacher/*"
    element={
        <PrivateRoute roles={["teacher"]}>
            <TeacherLayout />
        </PrivateRoute>
    }
>
    <Route
        index
        element={<Navigate to="home" replace />}
    />

    <Route path="home" element={<FileForm />} />

    <Route path="records" element={<Records />} />

    <Route path="syllabus" element={<AddSyllabus />} />

    <Route path="marks-upload" element={<MarksUpload />} />

    {/* ⭐ NEW: Mentorship Section */}
    <Route path="mentorship" element={<Mentorship />} />

</Route>


                    <Route
                        path="/student"
                        element={
                            <PrivateRoute roles={["student"]}>
                                <StudentDashboard />
                            </PrivateRoute>
                        }
                    />
                    <Route
                        path="/admin"
                        element={
                            <PrivateRoute roles={["admin"]}>
                                <ManageUsers />
                            </PrivateRoute>
                        }
                    />

                    <Route
                        path="/"
                        element={
                            <Navigate
                                to="/login"
                                replace
                            />
                        }
                    />
                    <Route
                        path="*"
                        element={<div>Not Found</div>}
                    />
                </Routes>
            </Router>
        </AuthProvider>
    );
}
export default App;
