import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("mnist_ann_model.h5")

def predict_digit(image):
    if image is None:
        return "Draw a digit first!"

    img_array = image["composite"]   # Sketchpad returns a dict, same as your earlier digits app

    img = Image.fromarray(img_array).convert("L")   # grayscale
    img = img.resize((28, 28))                        # match MNIST's real size this time, not 8x8

    pixels = np.array(img)
    pixels = 255 - pixels                              # invert: sketchpad draws dark-on-light, MNIST is light-on-dark
    pixels = pixels / 255.0                             # normalize, same as training
    pixels = pixels.reshape(1, 784)                     # flatten to match input_dim=784

    prediction = model.predict(pixels)[0]                # returns 10 probabilities (softmax output)
    predicted_digit = np.argmax(prediction)               # pick the highest-probability class
    confidence = prediction[predicted_digit]

    return f"Predicted digit: {predicted_digit} (confidence: {confidence:.2%})"

demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(),
    outputs="text",
    title="✍️ MNIST Digit Classifier (Neural Network)",
    description="Draw a digit (0-9) and the ANN will predict it.",
    theme=gr.themes.Soft()
)

demo.launch()