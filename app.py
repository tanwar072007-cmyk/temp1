
import gradio as gr
import joblib
import os

deployed_lr = joblib.load("my_first_ml_model.pkl")

def predict_rent(size_of_prop):
    prediction = deployed_lr.predict([[size_of_prop]])
    return f"Estimated Rent: {prediction[0]:.2f}"

interface = gr.Interface(
    fn=predict_rent,
    inputs=gr.Number(label="Please Enter the Size of Your Property for rent"),
    outputs=gr.Text(label="Predicted Rent"),
    title="Property Rent Predictor",
    description="Enter the property size to get a rent estimate powered by Machine Learning."
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
