import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Download } from 'lucide-react';
import * as Papa from 'papaparse';

const UploadCSV = ({ studentData, setStudentData, calculateRisk }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadedCount, setUploadedCount] = useState(0);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.name.endsWith('.csv')) {
      setUploadStatus({ type: 'error', message: 'Please upload a CSV file' });
      return;
    }

    setUploadStatus({ type: 'loading', message: 'Processing file...' });

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          const validData = results.data.filter(row => 
            row.name && row.math && row.science && row.english && row.attendance
          );

          if (validData.length === 0) {
            setUploadStatus({ type: 'error', message: 'No valid student data found' });
            return;
          }

          const processedData = validData.map(student => ({
            ...student,
            risk: calculateRisk(student)
          }));

          setStudentData([...studentData, ...processedData]);
          setUploadedCount(processedData.length);
          setUploadStatus({ 
            type: 'success', 
            message: `Successfully uploaded ${processedData.length} students` 
          });
        } catch (error) {
          setUploadStatus({ type: 'error', message: 'Error processing file' });
        }
      },
      error: () => {
        setUploadStatus({ type: 'error', message: 'Error reading file' });
      }
    });
  };

  const downloadSampleCSV = () => {
    const sampleData = [
      'name,math,science,english,attendance',
      'John Doe,85,78,92,95',
      'Jane Smith,72,89,84,88',
      'Mike Johnson,65,70,75,82'
    ].join('\n');

    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_students.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
          <h1 className="text-2xl font-bold text-gray-800 mb-2">Upload Student Data</h1>
          <p className="text-gray-600">Upload a CSV file containing student information to analyze performance and risk levels.</p>
        </div>

        {/* Upload Area */}
        <div className="bg-white rounded-xl p-8 shadow-sm border border-gray-200">
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
              dragActive 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Upload className={`mx-auto h-16 w-16 mb-4 ${dragActive ? 'text-blue-500' : 'text-gray-400'}`} />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">
              {dragActive ? 'Drop your CSV file here' : 'Upload CSV File'}
            </h3>
            <p className="text-gray-500 mb-6">Drag and drop your file here, or click to browse</p>
            
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
              id="csv-upload"
            />
            <label
              htmlFor="csv-upload"
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors cursor-pointer"
            >
              <FileText className="mr-2" size={20} />
              Choose File
            </label>
          </div>

          {/* Status Messages */}
          {uploadStatus && (
            <div className={`mt-6 p-4 rounded-lg flex items-center ${
              uploadStatus.type === 'success' ? 'bg-green-50 text-green-800' :
              uploadStatus.type === 'error' ? 'bg-red-50 text-red-800' :
              'bg-blue-50 text-blue-800'
            }`}>
              {uploadStatus.type === 'success' && <CheckCircle className="mr-2" size={20} />}
              {uploadStatus.type === 'error' && <AlertCircle className="mr-2" size={20} />}
              {uploadStatus.type === 'loading' && (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-2"></div>
              )}
              <span className="font-medium">{uploadStatus.message}</span>
            </div>
          )}
        </div>

        {/* Instructions & Sample */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
              CSV Format Requirements
            </h3>
            <div className="space-y-3 text-sm text-gray-600">
              <div className="flex items-center">
                <CheckCircle size={16} className="text-green-500 mr-2" />
                <span><strong>Columns:</strong> name, math, science, english, attendance</span>
              </div>
              <div className="flex items-center">
                <CheckCircle size={16} className="text-green-500 mr-2" />
                <span><strong>Grades:</strong> Numeric values (0-100)</span>
              </div>
              <div className="flex items-center">
                <CheckCircle size={16} className="text-green-500 mr-2" />
                <span><strong>Attendance:</strong> Percentage (0-100)</span>
              </div>
              <div className="flex items-center">
                <CheckCircle size={16} className="text-green-500 mr-2" />
                <span><strong>Headers:</strong> First row must contain column names</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              Download Sample
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Not sure about the format? Download our sample CSV file to see the expected structure.
            </p>
            <button
              onClick={downloadSampleCSV}
              className="inline-flex items-center px-4 py-2 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition-colors"
            >
              <Download className="mr-2" size={16} />
              Download Sample CSV
            </button>
          </div>
        </div>

        {/* Recent Uploads */}
        {uploadedCount > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Upload Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-2xl font-bold text-blue-600">{uploadedCount}</p>
                <p className="text-sm text-gray-600">Students Added</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-2xl font-bold text-green-600">{studentData.length}</p>
                <p className="text-sm text-gray-600">Total Students</p>
              </div>
              <div className="bg-red-50 p-4 rounded-lg">
                <p className="text-2xl font-bold text-red-600">
                  {studentData.filter(s => s.risk === 'High').length}
                </p>
                <p className="text-sm text-gray-600">High Risk Students</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadCSV;