Simple Real-time Sound Listener 
Hey there! This is a small Python project that listens to your microphone and tries to figure out what sounds are happening in real-time. It uses a lightweight machine learning model to classify what it hears.

What's Included
You'll find these three essential files:

detect_audio.py: The main Python script that handles microphone input and runs the classification.

soundclassifier_with_metadata.tflite: This is the brain of the project! It's the small, pre-trained model that does the sound classification.

labels.txt: A simple list of all the sounds the model knows how to recognize.

Getting Started (Setup)
To make this work on your computer, you'll need a few things:

Python: Make sure you have Python installed (version 3.x is best).

A Microphone: The script needs something to listen with!

Once you have Python, open your terminal and install the required libraries. This project uses TensorFlow Lite, NumPy, and Sounddevice for audio input.

Bash

pip install tensorflow numpy sounddevice
Note: If you run into issues with sounddevice, you might need to install a system-level audio library (like PortAudio). Check the python-sounddevice documentation for troubleshooting on your specific operating system.

How to Run It
Make sure all three files are in the same folder.

Open your terminal or command prompt and navigate to that folder.

Run the main script:

Bash

python detect_audio.py
The script will start listening, and every few seconds, it will print out the sound it thinks it hears! Press Ctrl+C to stop the script.

What Sounds Does It Know?
The model can currently recognize the categories listed in your labels.txt file. Based on your uploaded file, it looks like it can distinguish between:

  Background Noise

  iphone ringtone

  vivo ringtone

Feel free to swap out the model and labels to teach it to listen for anything you want!
