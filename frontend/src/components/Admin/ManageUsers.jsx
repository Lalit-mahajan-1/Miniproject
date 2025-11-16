// frontend/src/Components/Admin/ManageUsers.jsx
import { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from '../Navbar';
import './ManageUsers.css';

const ManageUsers = () => {
  const [users, setUsers] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    name: '', email: '', password: '', role: 'student', studentId: ''
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const res = await axios.get(`${import.meta.env.VITE_API_URL}/api/auth/users`);
      setUsers(res.data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');

    try {
      await axios.post(`${import.meta.env.VITE_API_URL}/api/auth/register`, formData);
      setMessage('User created successfully!');
      setFormData({ name: '', email: '', password: '', role: 'student', studentId: '' });
      setShowForm(false);
      fetchUsers();
    } catch (error) {
      setMessage(error.response?.data?.message || 'Error creating user');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to delete this user?')) return;

    try {
      await axios.delete(`${import.meta.env.VITE_API_URL}/api/auth/users/${id}`);
      setMessage('User deleted successfully');
      fetchUsers();
    } catch (error) {
      setMessage('Error deleting user');
    }
  };

  return (
    <>
      <Navbar />
      <div className="admin-container">
        <div className="admin-header">
          <h1>User Management</h1>
          <button onClick={() => setShowForm(!showForm)} className="add-user-btn">
            {showForm ? 'Cancel' : '+ Add User'}
          </button>
        </div>

        {message && <div className="message">{message}</div>}

        {showForm && (
          <div className="user-form-card">
            <h2>Create New User</h2>
            <form onSubmit={handleSubmit} className="user-form">
              <input
                type="text"
                placeholder="Full Name"
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                required
              />
              <input
                type="email"
                placeholder="Email"
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
                required
              />
              <input
                type="password"
                placeholder="Password"
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                required
              />
              <select
                value={formData.role}
                onChange={(e) => setFormData({...formData, role: e.target.value})}
              >
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
                <option value="admin">Admin</option>
              </select>
              {formData.role === 'student' && (
                <input
                  type="text"
                  placeholder="Student ID"
                  value={formData.studentId}
                  onChange={(e) => setFormData({...formData, studentId: e.target.value})}
                  required
                />
              )}
              <button type="submit" disabled={loading}>
                {loading ? 'Creating...' : 'Create User'}
              </button>
            </form>
          </div>
        )}

        <div className="users-table">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Role</th>
                <th>Student ID</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map(user => (
                <tr key={user._id}>
                  <td>{user.name}</td>
                  <td>{user.email}</td>
                  <td><span className={`role-badge ${user.role}`}>{user.role}</span></td>
                  <td>{user.studentId || '-'}</td>
                  <td>
                    <button onClick={() => handleDelete(user._id)} className="delete-btn">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
};

export default ManageUsers;