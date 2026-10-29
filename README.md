MNIST Digit Chatbot (No CNN)

A fully connected neural network (MLP) trained only on the MNIST dataset, plus a terminal chatbot that uses the saved model to recognize handwritten digits.

Project structure
project/
├── script2.py        # trains the model and saves it
├── script3.py        # chatbot that loads the saved model
├── saved_model1/     # created by script2.py
│   └── mnist_mlp.keras
└── README.md
Requirements
Python 3.9+
Install the packages:
pip install tensorflow pillow numpy
How to run

1. Train the model (script2.py)

python script2.py
Loads MNIST (downloaded automatically on first run)
Trains a Dense-only network: Flatten → Dense(256) → Dense(128) → Dense(10)
Prints test accuracy (about 97-98%)
Saves the model to saved_model1/mnist_mlp.keras

2. Start the chatbot (script3.py)

python script3.py

Run script2.py first, otherwise the chatbot will say the model was not found.

Chatbot commands
Input	What it does
hi, hello, hey	Greeting
help or ?	Shows the commands
test	Predicts a random image from the MNIST test set
image path (e.g. my_digit.png)	Predicts the digit in your image
quit, exit, bye	Exits the chatbot
Example session
Digit Bot: Hi! Type 'help' for commands, 'quit' to exit.
You: test
Digit Bot: I think it's a 7 (99.8% sure). The real label is 7, so I was correct.
You: my_digit.png
Digit Bot: That looks like a 3 (94.5% sure).
You: quit
Digit Bot: Goodbye!
Tips for your own digit images
Draw one digit, large and centered, in a dark pen on white paper (or a thick brush in Paint).
Save as PNG or JPG in the same folder as script3.py.
The bot auto-inverts light backgrounds and resizes to 28x28.
Thin lines, multiple digits, or cluttered backgrounds reduce accuracy.
Notes
The network uses only fully connected (Dense) layers, with no convolutional layers.
MNIST contains only handwritten digit images, so the chatbot can recognize digits but cannot hold general conversations.
