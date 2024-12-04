#Note:How to use this code is mentioned at end of this code ensure we proceed in that order for better results.
#Also what will be the output in output window will also be mentioned there in a step by step manner.

#Please Ensure adding correct path of sound files.


#importing all neccessary files for this project.
import cv2  # for computer vision and image processing techniques
import mediapipe as mp  # for hand-tracking
import pygame  # for playing audio files when gesture detected
import time  # to work with timestamps and delays
import os  # to interact with file system
import numpy as np  # for numerical computations

pygame.mixer.init()  # for loading and playing audio files

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # loads the relative sound file paths irrespective of different computer systems

#Specifies the relative file paths for the audio files associated with each gesture.
sound_files = {
    'thumbs_up': os.path.join(BASE_DIR, 'SoundFiles', 's11.wav'),
    'index_up': os.path.join(BASE_DIR, 'SoundFiles', 's22.wav'),
    'middle_up': os.path.join(BASE_DIR, 'SoundFiles', 's33.wav'),
    'ring_up': os.path.join(BASE_DIR, 'SoundFiles', 's44.wav'),
    'pinky_up': os.path.join(BASE_DIR, 'SoundFiles', 's55.wav')
}# .wav files for reliable and fast playback


# Validates if each sound file exists,otherwise,the program exits.
sounds = {}
for gesture, path in sound_files.items():
    if not os.path.exists(path):
        print(f"Sound file for {gesture} not found: {path}")
        exit()
    sounds[gesture] = pygame.mixer.Sound(path)
# Loads each audio file into a pygame Sound object and stores it in the sounds dictionary.


#mp_hands:Provides access to MediaPipe's hand detection functionality.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, #Specifies the maximum number of hands to detect in the video frame.
    min_detection_confidence=0.8,  # Ensures the model only detects hands when it's reasonably confident.
    min_tracking_confidence=0.8  # Ensures reliable tracking of hand movement across frames.
)
mp_draw = mp.solutions.drawing_utils  # This utility is used to draw hand landmarks and connections.


#This function plays the sound corresponding to the detected gesture.
def play_sound(gesture):
    if gesture in sounds:
        sounds[gesture].play()


#This function stops all currently playing sounds.
def stop_all_sounds():
    for snd in sounds.values():
        snd.stop()


#This function returns a 5 length list to detect which finger is up.
def fingers_up(hand_landmarks):# hand_landmarks is mediapipe object containing x,y,z coordinates of detected hand.
    fingers = [] #5 length list which this function will return.
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
#A list of the fingertip landmarks for the thumb, index, middle, ring, and pinky fingers, provided by MediaPipe.
#These landmarks correspond to specific points on the hand model.

    for tip in finger_tips:
#The thumb’s orientation is different from the other fingers, as it extends horizontally rather than vertically.
        if tip == mp_hands.HandLandmark.THUMB_TIP:
            if hand_landmarks.landmark[tip].x > hand_landmarks.landmark[tip - 1].x:
                fingers.append(1) #Thumb is extended (1)
            else:
                fingers.append(0) #Thumb is not extended (0)
        elif hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1) #Finger is extended (1)
        else:
            fingers.append(0) #Finger is not extended (0)

    return fingers
#returns a list named fingers having five elements and only one (1) at a time and remaining four as (0).  


#this function blur the input image,here we are using box kernel of size 5*5.
#Simply creating a 5*5 2-D list and convolve with input image. 
def apply_optimized_blur(image, kernel_size=5):
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
#This creates a normalized kernel of size kernel_size*kernel_size having all elements
# as 1 divided by number of elements i.e. (kernel_size)^2. 

    pad = kernel_size // 2
# Calculate the amount of padding required to ensure that the convolution operation doesn't shrink the image 
# i.e output image has the same size as the input image after convolution.
# here kernel_size is 5 so output will be 5//2=2 (// operator means taking float values.)

    image_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
#The input image with extra pixels added around the edges.
#mode mirror (also called reflective) padding, in which values outside the boundary of the image are 
#obtained by mirror-reflecting the image across its border.

    blurred = np.zeros_like(image, dtype=np.float32)
# Create output empty image.
   
#Convolution
#This is another way of doing convolution as we know (i,j) value of 
#kernel contribute to a particular window of the final values only.
    for c in range(3):  # For each color channel i.e R,G,B
        for i in range(kernel_size):
            for j in range(kernel_size):
                blurred[:, :, c] += kernel[i, j] * image_padded[i:i + image.shape[0], j:j + image.shape[1], c]

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
# np.clip(blurred, 0, 255): Ensures all pixel values are within the valid range [0, 255].
# Any value below 0 becomes 0, and any value above 255 becomes 255.
    
    return blurred


#This function returns binary image for the input image if pixel-value<=threshold  pixel of 
#output becomes 0(black/dark) else 255(white/light).
def apply_optimized_threshold(image, threshold=127):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Converts the input color image from BGR (Blue-Green-Red) to Grayscale.
#Grayscale images have a single channel, where each pixel’s intensity is a value between 0 (black) and 255 (white).

    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
# _: The threshold value used (not needed in this case, so it's ignored).
# binary_image: The resulting black-and-white (binary) image.

    return binary_image


#This function is used to measure the average brightness in the camera's view
def calibrate_environment(cap):
#cap an instance of cv2.VideoCapture, which manages the camera view.

    print("Please show your hand to the camera.")
    
    _, frame = cap.read()
# _: A boolean indicating whether the frame was captured successfully (ignored here).
# frame: The captured frame, which is an image in BGR format.

    avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
#Converts the captured BGR frame into a grayscale image then np.mean() will compute the averege brightness of input image.

    print(f"Average Brightness: {avg_brightness}")
    return avg_brightness


#This function is used to detect the background of camera's view.
def capture_background(cap):
    print("Please step out of the frame.")
#Step out of the frame so the background can be captured without any interference.
    time.sleep(3)
#A time delay of 3 sec to step out of the frame so that background can easily be detected. 

    _, background = cap.read()
    print("Background captured.")
    return background
#This function is used for detecting gestures by comparing the current frame to the background frame.


#This function adapts the meaning of gestures based on the brightness of the environment.
def adapt_gesture_meaning(avg_brightness):
    if avg_brightness < 50:
        print("Low light detected Software can give unexpected results.")
        return {
    'thumbs_up': 'Thumbs up detected playing Sound',
    'index_up': 'Index finger up detected playing Sound',
    'middle_up': 'Middle finger up detected playing Sound',
    'ring_up': 'Ring finger up detected playing Sound',
    'pinky_up': 'Pinky finger up detected playing Sound'
          }
    else:
        return {
    'thumbs_up': 'Thumbs up detected playing Sound',
    'index_up': 'Index finger up detected playing Sound',
    'middle_up': 'Middle finger up detected playing Sound',
    'ring_up': 'Ring finger up detected playing Sound',
    'pinky_up': 'Pinky finger up detected playing Sound'
        }


#frame by frame processing the video and giving the expected results.
def process_video():
    cap = cv2.VideoCapture(0) #open webcam 
    if not cap.isOpened(): #if we cannot open webcam then print error message
        print("ERROR!:Cannot open the webcam")
        return

    avg_brightness = calibrate_environment(cap) 
    #Calculating the average brightness of frame 

    background = capture_background(cap)
    #detecting the background

    gesture_meanings = adapt_gesture_meaning(avg_brightness)
    #meaning of every gesture

    gesture_detected = None
#storing which gesture was previously detected.

    #moving frame by frame on instnce of video i.e. cap
    while cap.isOpened():
        success, img = cap.read()
        if not success: #if .read() is unable to read print error message
            print("ERROR!:No frame detected")
            break

#blurring the input image for noise reduction and improving object detection.
        blurred = apply_optimized_blur(img)
        threshold = apply_optimized_threshold(img)

        img_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
#converting image from bgr format to rgb format because mediapipe expects rgb images. 
        
        results = hands.process(img_rgb)
#The hands.process function is used to process the frame and detect hand landmarks.

        current_gesture = None
#storing currently detected gesture.

        if results.multi_hand_landmarks:  # Checks if any hand landmarks are detected in the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
# If any hands are detected in the frame (results.multi_hand_landmarks contains detected landmarks),
# this loop iterates over each hand. It then draws the landmarks on the frame using mp_draw.draw_landmarks 
# and determines the finger states (whether each finger is up or down) by calling the fingers_up function.

                finger_states = fingers_up(hand_landmarks)

                if finger_states == [1, 0, 0, 0, 0]:
                    current_gesture = 'thumbs_up'
                elif finger_states == [0, 1, 0, 0, 0]:
                    current_gesture = 'index_up'
                elif finger_states == [0, 0, 1, 0, 0]:
                    current_gesture = 'middle_up'
                elif finger_states == [0, 0, 0, 1, 0]:
                    current_gesture = 'ring_up'
                elif finger_states == [0, 0, 0, 0, 1]:
                    current_gesture = 'pinky_up'
                else:
                    current_gesture = None
#Handling all five cases and one none if no current gesture is detected.

#Breaks the loop if a gesture is recognized to prioritize the first detected hand.
                if current_gesture:
                    break

#if the current gesture is differnet from previous one then 
#retrive the meaning of the gesture by .get operation then playing the sound corresponding to current gesture.
        if current_gesture != gesture_detected:
            if current_gesture:# to avoid none in current gesture.
                print(f"{gesture_meanings.get(current_gesture, current_gesture)}!")
                play_sound(current_gesture)
                gesture_detected = current_gesture
#now for next frame current gesture will be previously detected gesture.

            else:
                if gesture_detected:#to avoid none in previously detected gesture.
                    print("Gesture ended")
                    stop_all_sounds()
                    gesture_detected = None

#Horizontally stacking the 2 images first original+blurred image.
        combined_view = np.hstack((img, blurred))

#showing binary image in another window.
        cv2.imshow("Threshold", threshold)

#showing the combined view of image and its blurred image in one window.
        cv2.imshow("Hand Gesture-Controlled Sound Synthesis:Original(Left) , Blurred(Right)", combined_view)

#if letter q will be pressed all windows and webcam will be closed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#release all the acquired resources like webcam.
    cap.release()

#destory all opened windows.
    cv2.destroyAllWindows()


#main function to which everyone will be reporting.
def main():
    process_video()


if __name__ == "__main__":
    main()

#WorkFlow:
#1.
# Run the given code.
#2.
# In output window we get: Please show your hand to camera OR Error messsage saying cannot open the web cam 
# in later option re run the code if issue still there please clear the terminal then re run the code.
#3.
# Now average brightness of the captured image will be shown if
# average brightness is less than 50 ensure lighting the room as output window shows 
# Low light detected Software can give unexpected results.
#4.
# Now a message of please step out of the frame will be displayed ensure stepping out of frame in 3 seconds
# as a delay of 3 second is given.Then This will capture the background image showing Background Captured.
#5.
# directions in which axis are aligned is:
# increasing + values in left side(+x-axiz) and 
# increasing + values in down side(+y-axis). 
#6.
# So orient your thumb on left side implying for right hand, palm side should be shown while for 
# left hand back side should be shown so that thumb oriented towards left.
#7.
# Now show only 1 finger at a time the code will play corresponding sound.
# to switch between fingers first show all fingers then do the same step. 
# So output window will look like:
# Thumbs up detected playing Sound!                 (Showing Thumb) 
# Gesture ended                                     (Showing all fingers)
# Pinky finger up detected playing Sound!           (Showing Pinky finger)
# Gesture ended                                     (Showing all fingers)
# Ring finger up detected playing Sound!            (Showing ring finger)
# Gesture ended                                     (Showing all fingers)

#We proceed in this way to not confuse the code in playing sounds.

#6 For end click Q key.
