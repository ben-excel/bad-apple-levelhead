import cv2
import os

# Function to process video frames and export brightness information to a text file
def process_video(video_filename, output_filename):
    video_path = os.path.join(os.getcwd(), video_filename)
    cap = cv2.VideoCapture(video_path)
    brightness_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to grayscale for better brightness analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze brightness per pixel
        brightness_row = ""
        for y in range(gray_frame.shape[0]):
            for x in range(gray_frame.shape[1]):
                pixel_brightness = gray_frame[y, x]
                
                # Set threshold for brightness (adjust as needed)
                threshold = 127
                if pixel_brightness > threshold:
                    brightness_row += "1"
                else:
                    brightness_row += "0"
        
        brightness_data.append(brightness_row)
    
    cap.release()
    
    # Write the brightness data into a text file
    with open(output_filename, 'w') as file:
        for i, row in enumerate(brightness_data):
            if i != len(brightness_data) - 1:  # Check if it's not the last row
                file.write(f'"{row}",\n')
            else:
                file.write(f'"{row}"')

# Provide the filename of the video in the current directory
video_filename = "Bad Apple (low res).mp4"
output_filename = "data.txt"
process_video(video_filename, output_filename)
