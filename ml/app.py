from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from utils import StudentRiskPredictor

app = FastAPI(title="Student Risk Prediction API", version="1.0.0")

# Initialize the risk predictor
risk_predictor = StudentRiskPredictor()

class Student(BaseModel):
    name: str
    math: float
    science: float
    english: float
    attendance: float

class StudentBulk(BaseModel):
    students: List[Student]

@app.get("/")
async def root():
    return {"message": "Student Risk Prediction API", "status": "running"}

@app.post("/predict-risk")
async def predict_risk(student: Student):
    """Predict risk level for a single student"""
    try:
        risk_data = risk_predictor.predict_single(
            student.math, student.science, student.english, student.attendance
        )
        
        return {
            "success": True,
            "student": student.name,
            "prediction": risk_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-risk-bulk")
async def predict_risk_bulk(data: StudentBulk):
    """Predict risk levels for multiple students"""
    try:
        predictions = []
        
        for student in data.students:
            risk_data = risk_predictor.predict_single(
                student.math, student.science, student.english, student.attendance
            )
            predictions.append({
                "name": student.name,
                "risk": risk_data["risk_level"],
                "probability": risk_data["probability"],
                "factors": risk_data["risk_factors"]
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "total_students": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics")
async def get_analytics(data: StudentBulk):
    """Get comprehensive analytics for student data"""
    try:
        analytics = risk_predictor.analyze_cohort(data.students)
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)