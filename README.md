Project Goal:
The aim of the project is to create an innovative virtual musical instrument that uses real-time hand gestures for sound control, enhancing user interaction with digital audio environments. By using real-time hand tracking and gesture recognition, the project seeks to create an intuitive interface for sound synthesis, providing a novel way to interact with digital music production tools.

Methodology: The project is divided into three main phases:
1.Hand Tracking:
Hand tracking is achieved using MediaPipe, a powerful and efficient library for real-time hand landmark detection. The hand positions are tracked via a webcam feed, and key landmarks on the hands are identified.
2.Gesture Recognition:
Specific hand gestures, such as "thumbs up," are recognized by analyzing the relative positions of the detected hand landmarks.
3.Sound Synthesis:
The recognized gestures are then mapped to sound synthesis using the Pyo library. For each detected gesture, a corresponding sound is generated in real-time. The "thumbs up" gesture, for example, triggers a sine wave sound. This basic mapping can be extended to incorporate more gestures and complex sound controls.

Tech Stack:
1.Programming Language: Python
2.Libraries: OpenCV for Video capture and processing, MediaPipe for Hand tracking, Pyo for Sound synthesis
3.Hardware: Webcam and PC/Laptop

Applications:
1.Music Production: Develop gesture-controlled virtual instruments for creative music-making.
2.Assistive Technology: Provide a new method of music creation for individuals with physical disabilities.

Project Contributors:
Raj Gupta(22115126)                Tanishq Garg(22116093)
Swatantra Dwivedi(22116092)        Arun Kumar(22116016)
