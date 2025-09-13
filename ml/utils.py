import numpy as np
from typing import List, Dict, Any

class StudentRiskPredictor:
    def __init__(self):
        # Risk thresholds based on educational standards
        self.grade_threshold_high = 60
        self.grade_threshold_medium = 75
        self.attendance_threshold_high = 75
        self.attendance_threshold_medium = 85
    
    def calculate_average_grade(self, math: float, science: float, english: float) -> float:
        """Calculate average grade across subjects"""
        return (math + science + english) / 3
    
    def predict_single(self, math: float, science: float, english: float, attendance: float) -> Dict[str, Any]:
        """Predict risk level for a single student"""
        avg_grade = self.calculate_average_grade(math, science, english)
        
        # Risk calculation logic
        risk_score = 0
        risk_factors = []
        
        # Grade-based risk
        if avg_grade < self.grade_threshold_high:
            risk_score += 0.6
            risk_factors.append("Low academic performance")
        elif avg_grade < self.grade_threshold_medium:
            risk_score += 0.3
            risk_factors.append("Below average academic performance")
        
        # Attendance-based risk
        if attendance < self.attendance_threshold_high:
            risk_score += 0.4
            risk_factors.append("Poor attendance")
        elif attendance < self.attendance_threshold_medium:
            risk_score += 0.2
            risk_factors.append("Moderate attendance issues")
        
        # Subject-specific risks
        subjects = {"Math": math, "Science": science, "English": english}
        failing_subjects = [subject for subject, score in subjects.items() if score < 60]
        if failing_subjects:
            risk_factors.append(f"Failing in: {', '.join(failing_subjects)}")
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "High"
            probability = min(0.9, 0.6 + risk_score * 0.3)
        elif risk_score >= 0.3:
            risk_level = "Medium"
            probability = min(0.7, 0.3 + risk_score * 0.4)
        else:
            risk_level = "Low"
            probability = max(0.1, risk_score * 0.3)
        
        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "probability": round(probability, 2),
            "average_grade": round(avg_grade, 1),
            "risk_factors": risk_factors,
            "recommendations": self.get_recommendations(risk_level, risk_factors)
        }
    
    def get_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get recommendations based on risk level and factors"""
        recommendations = []
        
        if risk_level == "High":
            recommendations.extend([
                "Schedule immediate parent-teacher conference",
                "Consider additional tutoring support",
                "Monitor daily progress closely"
            ])
        elif risk_level == "Medium":
            recommendations.extend([
                "Provide additional learning resources",
                "Monitor weekly progress",
                "Consider study group participation"
            ])
        else:
            recommendations.append("Continue current learning pace")
        
        # Factor-specific recommendations
        if any("attendance" in factor.lower() for factor in risk_factors):
            recommendations.append("Address attendance issues with family")
        
        if any("failing" in factor.lower() for factor in risk_factors):
            recommendations.append("Focus on subject-specific interventions")
        
        return recommendations
    
    def analyze_cohort(self, students: List[Any]) -> Dict[str, Any]:
        """Analyze entire student cohort"""
        if not students:
            return {"error": "No student data provided"}
        
        predictions = []
        for student in students:
            pred = self.predict_single(
                student.math, student.science, student.english, student.attendance
            )
            predictions.append(pred)
        
        # Cohort statistics
        risk_distribution = {
            "High": len([p for p in predictions if p["risk_level"] == "High"]),
            "Medium": len([p for p in predictions if p["risk_level"] == "Medium"]),
            "Low": len([p for p in predictions if p["risk_level"] == "Low"])
        }
        
        avg_scores = {
            "math": np.mean([s.math for s in students]),
            "science": np.mean([s.science for s in students]),
            "english": np.mean([s.english for s in students]),
            "attendance": np.mean([s.attendance for s in students])
        }
        
        return {
            "total_students": len(students),
            "risk_distribution": risk_distribution,
            "average_scores": {k: round(v, 1) for k, v in avg_scores.items()},
            "high_risk_percentage": round((risk_distribution["High"] / len(students)) * 100, 1),
            "class_recommendations": self.get_class_recommendations(risk_distribution, len(students))
        }
    
    def get_class_recommendations(self, risk_dist: Dict[str, int], total: int) -> List[str]:
        """Get class-level recommendations"""
        recommendations = []
        high_risk_pct = (risk_dist["High"] / total) * 100
        
        if high_risk_pct > 30:
            recommendations.append("Consider curriculum review and teaching method adjustments")
        if high_risk_pct > 20:
            recommendations.append("Implement class-wide intervention programs")
        if risk_dist["High"] > 0:
            recommendations.append("Schedule individual meetings with high-risk students")
        
        return recommendations