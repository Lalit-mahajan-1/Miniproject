// Modern Tailwind Login Page
import { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../../context/AuthContext";
import { FiLogIn, FiMail, FiLock } from "react-icons/fi";

const Login = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const { login } = useContext(AuthContext);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setLoading(true);

        try {
            const user = await login(email, password);

            if (user.role === "admin") navigate("/admin");
            else if (user.role === "teacher") navigate("/teacher");
            else navigate("/student");
        } catch (err) {
            setError(err.response?.data?.message || "Login failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col justify-between">
            {/* Header */}
            <header className="w-full py-6 bg-white shadow-sm">
                <h1 className="text-2xl font-bold text-center text-gray-800 tracking-tight">Student Risk Prediction System</h1>
            </header>

            {/* Main Login Card */}
            <div className="flex justify-center items-center flex-1 px-4">
                <div className="bg-white w-full max-w-md shadow-xl rounded-2xl p-8 border border-gray-100">
                    <h2 className="text-3xl font-semibold text-gray-800 mb-2 text-center">Welcome Back</h2>
                    <p className="text-gray-500 text-center mb-6">Sign in to continue</p>

                    <form
                        onSubmit={handleSubmit}
                        className="space-y-5"
                    >
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                            <div className="relative">
                                <FiMail className="absolute left-3 top-3 text-gray-400" />
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                    placeholder="Enter your email"
                                    className="w-full pl-10 pr-4 py-2.5 border border-neutral-400 rounded-lg bg-gray-50 focus:bg-white focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                            <div className="relative">
                                <FiLock className="absolute left-3 top-3 text-gray-400" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    placeholder="Enter your password"
                                    className="w-full pl-10 pr-4 py-2.5 border border-neutral-400 rounded-lg bg-gray-50 focus:bg-white focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                                />
                            </div>
                        </div>

                        {error && <div className="text-red-600 bg-red-100 p-2 rounded text-sm text-center">{error}</div>}

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 transition text-white py-3 rounded-lg font-semibold shadow-md"
                        >
                            <FiLogIn />
                            {loading ? "Signing in..." : "Sign In"}
                        </button>
                    </form>
                </div>
            </div>

            {/* Footer */}
            <footer className="py-4 text-center text-gray-500 text-sm">
                © {new Date().getFullYear()} Student Risk Predictor • All Rights Reserved
            </footer>
        </div>
    );
};

export default Login;
