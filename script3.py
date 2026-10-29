import os
import random
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
MODEL_PATH = os.path.join("saved_model", "mnist_mlp.keras")
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run train_mnist.py first.")
model = keras.models.load_model(MODEL_PATH)
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
def predict_array(img28):
    probs = model.predict(img28.reshape(1, 28, 28), verbose=0)[0]
    digit = int(np.argmax(probs))
    return digit, float(probs[digit])
def load_image(path):
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    return np.array(img, dtype="float32") / 255.0
def respond(text):
    t = text.strip()
    low = t.lower()
    if low in ("hi", "hello", "hey"):
        return "Hello! Send me the path to an image of a handwritten digit, or type 'test'."
    if low in ("help", "?"):
        return ("Commands:\n"
                "  <image path>  - I'll tell you which digit is in the image\n"
                "  test          - I'll try a random digit from the MNIST test set\n"
                "  quit          - exit")
    if low == "test":
        i = random.randrange(len(x_test))
        digit, conf = predict_array(x_test[i] / 255.0)
        verdict = "correct" if digit == int(y_test[i]) else "wrong"
        return (f"I think it's a {digit} ({conf:.1%} sure). "
                f"The real label is {y_test[i]}, so I was {verdict}.")
    if os.path.isfile(t):
        try:
            digit, conf = predict_array(load_image(t))
            return f"That looks like a {digit} ({conf:.1%} sure)."
        except Exception as e:
            return f"I couldn't read that image: {e}"
    return "I didn't get that. Type 'help' to see what I can do."
def main():
    print("Digit Bot: Hi! Type 'help' for commands, 'quit' to exit.")
    while True:
        user = input("You: ")
        if user.strip().lower() in ("quit", "exit", "bye"):
            print("Digit Bot: Goodbye!")
            break
        print("Digit Bot:", respond(user))
if __name__ == "__main__":
    main()