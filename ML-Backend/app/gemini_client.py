import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        result = model.generate_content(prompt)
        return result.text
    except Exception as e:
        print("🔥 GEMINI ERROR:", e)
        return f"Gemini API error: {e}"
