import gradio as gr
import numpy as np
import pickle
import os

MODEL_PATH = "model (2).pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

def predict_weather(temperature, humidity, wind_speed, pressure, cloud_cover, visibility):
    features = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover, visibility]])
    
    if model is not None:
        try:
            prediction = int(model.predict(features)[0])
            try:
                proba = model.predict_proba(features)[0]
                confidence = round(float(max(proba)) * 100, 1)
            except:
                confidence = None
        except:
            prediction = None
            confidence = None
    else:
        prediction = None
        confidence = None

    if prediction is None:
        score = 0
        if humidity > 70: score += 2
        if humidity > 85: score += 2
        if cloud_cover > 60: score += 2
        if wind_speed > 25: score += 1
        if pressure < 1005: score += 1
        if visibility < 5: score += 1
        prediction = 1 if score >= 4 else 0

    if prediction == 1:
        result = "🌧️ RAIN EXPECTED"
        advice = "☂️ Carry an umbrella! Rain is likely today."
    else:
        result = "☀️ NO RAIN - Clear Sky"
        advice = "😎 Enjoy the sunshine! No rain expected."

    conf = f"\nConfidence: {confidence}%" if confidence else ""
    return f"{result}{conf}\n\n{advice}"

with gr.Blocks(title="Weather Prediction") as demo:
    gr.Markdown("# 🌦️ Weather Prediction App")
    gr.Markdown("Enter weather conditions to predict rain.")
    
    with gr.Row():
        with gr.Column():
            temperature = gr.Slider(-10, 50,   value=25,   step=0.5, label="🌡️ Temperature (°C)")
            humidity    = gr.Slider(0,   100,  value=60,   step=1,   label="💧 Humidity (%)")
            wind_speed  = gr.Slider(0,   100,  value=15,   step=0.5, label="💨 Wind Speed (km/h)")
            pressure    = gr.Slider(950, 1050, value=1013, step=0.5, label="🔵 Pressure (hPa)")
            cloud_cover = gr.Slider(0,   100,  value=40,   step=1,   label="☁️ Cloud Cover (%)")
            visibility  = gr.Slider(0,   20,   value=10,   step=0.5, label="👁️ Visibility (km)")
            btn = gr.Button("🔍 Predict Weather")
        
        with gr.Column():
            output = gr.Textbox(label="🌦️ Result", lines=4)
    
    btn.click(
        fn=predict_weather,
        inputs=[temperature, humidity, wind_speed, pressure, cloud_cover, visibility],
        outputs=output
    )

demo.launch()
