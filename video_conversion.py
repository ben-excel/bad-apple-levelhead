import os
from moviepy.editor import VideoFileClip

def convert_video(input_filename, output_filename):
    # Get the current working directory
    current_directory = os.getcwd()

    # Define the full file paths for input and output
    input_path = os.path.join(current_directory, input_filename)
    output_path = os.path.join(current_directory, output_filename)

    # Load the video
    clip = VideoFileClip(input_path)
    
    # Resize the video to 16x12 pixels
    resized_clip = clip.resize((16, 12))

    # Set the frame rate to 3 fps
    final_clip = resized_clip.set_fps(3)

    # Save the modified video
    final_clip.write_videofile(output_path, codec='libx264')

# Replace 'Bad Apple.mp4' and 'Bad Apple (low res).mp4' with your file names
convert_video('Bad Apple.mp4', 'Bad Apple (low res).mp4')
