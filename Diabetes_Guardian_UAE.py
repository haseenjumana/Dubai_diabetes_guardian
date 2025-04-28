from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import json
import uvicorn


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load UAE diabetes knowledge base
with open("static/uae-clinics.json") as f:
    clinics = json.load(f)

# Load medical model with compilation to prevent warnings
diabetes_model = tf.keras.models.load_model("diabetes_model.h5", compile=False)
diabetes_model.compile(optimizer='adam', loss='binary_crossentropy')

def get_uae_advice(risk_level):
    advice = {
        "low": {
            "diet": ["Whole grains", "Dates (3-5/day)", "Grilled fish"],
            "checkups": "Annual DHA screening"
        },
        "medium": {
            "diet": ["Limit Arabic sweets", "Avoid sugary drinks"],
            "checkups": "Bi-annual HbA1c tests"
        },
        "high": {
            "diet": ["Consult DHA nutritionist", "Low-carb plan"],
            "checkups": "Monthly monitoring"
        }
    }
    return advice[risk_level]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request,
                 age: int = Form(...),
                 bmi: float = Form(...),
                 glucose: int = Form(...),
                 hba1c: float = Form(...)):
    
    # UAE-specific risk calculation
    input_data = np.array([[age, bmi, glucose, hba1c]], dtype=np.float32)
    prediction = diabetes_model.predict(input_data)
    risk = round(float(prediction[0][0]) * 100, 1)
    
    # Risk classification (WHO Middle East guidelines)
    if risk < 30:
        status = "Low Risk"
        category = "low"
    elif 30 <= risk < 60:
        status = "Pre-Diabetes"
        category = "medium"
    else:
        status = "High Risk"
        category = "high"
    
    # UAE-specific recommendations
    advice = get_uae_advice(category)
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "risk": risk,
        "status": status,
        "category": category,
        "advice": advice,
        "clinics": clinics,
        "emergency": "800-DHA (342)"
    })


if __name__ == "__main__":
    uvicorn.run("Diabetes_Guardian_UAE:app", host="0.0.0.0", port=8000)
