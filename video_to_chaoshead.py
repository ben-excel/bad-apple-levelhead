import cv2
import os
import numpy as np
import time
from moviepy.editor import VideoFileClip

# --- PART 1: VIDEO PREPARATION ---
# This first section handles converting the input video into a small, low-framerate
# version that will be used to generate the in-game display.

def convert_and_resize_video(input_filename, output_filename, target_width, target_height, target_fps):
    """
    Takes a standard video file and converts it into a small,
    low-resolution, and specific-framerate video using the moviepy library.

    Args:
        input_filename (str): The name of the video file to process (e.g., "video.mp4").
        output_filename (str): The name for the new, smaller video file.
        target_width (int): The desired width in pixels for the new video.
        target_height (int): The desired height in pixels for the new video.
        target_fps (int): The desired frames-per-second for the new video.
    Returns:
        bool: True if the conversion was successful, False if an error occurred.
    """
    # Get the full path to the video files in the current directory
    current_directory = os.getcwd()
    input_path = os.path.join(current_directory, input_filename)
    output_path = os.path.join(current_directory, output_filename)

    # Check if the input video actually exists before trying to open it.
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at {input_path}")
        return False

    print(f"Converting '{input_filename}' to a {target_width}x{target_height} video at {target_fps} FPS...")

    # Use a try...finally block to ensure video files are properly closed,
    # even if an error happens during processing.
    clip = None
    resized_clip = None
    final_clip = None
    try:
        # Load the video file into moviepy
        clip = VideoFileClip(input_path)
        # Resize the video to the small target dimensions
        resized_clip = clip.resize(newsize=(target_width, target_height))
        # Set the video's frame rate
        final_clip = resized_clip.set_fps(target_fps)
        
        # Write the final, processed video to a new file.
        # logger=None prevents moviepy from printing its own progress bar.
        # preset='ultrafast' and threads=4 help speed up the video encoding process.
        final_clip.write_videofile(output_path, codec='libx264', logger=None, threads=4, preset='ultrafast')
        
        print("Video conversion and resizing complete.")
        return True
    except Exception as e:
        print(f"An error occurred during video conversion: {e}")
        return False
    finally:
        # Clean up by closing all the video clip objects we opened.
        if clip:
            clip.close()
        if resized_clip:
            resized_clip.close()
        if final_clip:
            final_clip.close()

# --- PART 2: COLOR MAPPING ---
# This section defines the limited color palette we will use for the in-game display.
# Every pixel in the video will be converted to the closest-matching color from this palette.

# Define the 6 specific colors that our in-game "pixels" can be.
# Note: OpenCV uses a Blue-Green-Red (BGR) format, not the more common RGB.
palette_bgr = np.array([
    [28, 23, 24],   # 0: Black
    [53, 37, 253],  # 1: Red
    [0, 210, 253],  # 2: Yellow
    [98, 166, 0],   # 3: Green
    [215, 27, 37],  # 4: Blue
    [204, 204, 204] # 5: White
], dtype=np.uint8)

# These are the labels for our palette colors. '0' for Black, '1' for Red, etc.
palette_labels = np.array(['0', '1', '2', '3', '4', '5'])

def map_colors_to_palette_vectorized(frame, palette):
    """
    Takes a video frame and converts every pixel to its closest match in our color palette.
    This function uses numpy for high-speed mathematical operations to make this process very fast.
    """
    # Reshape the frame from a 2D image grid into a simple list of pixels
    pixels = frame.reshape(-1, 3).astype(np.float32)
    palette_float = palette.astype(np.float32)
    
    # Calculate the "distance" between each pixel's color and each palette color.
    # Using the squared distance is a common performance trick because it gives the
    # same result as a regular distance check but is faster to compute.
    distances_sq = np.sum((pixels[:, np.newaxis, :] - palette_float[np.newaxis, :, :])**2, axis=2)
    
    # For each pixel, find the palette color with the smallest distance (the closest match).
    indices = np.argmin(distances_sq, axis=1)
    return indices

# --- PART 3: LUA SCRIPT GENERATION ---
# This is the core of the script. It defines a template for a Lua script that can be
# run inside the game "Chaoshead" to build the video display system.

LUA_SCRIPT_TEMPLATE = """-- This Lua script is automatically generated by a Python script.
-- Its purpose is to build an in-game system that plays a video using
-- logic gates (relays) and colored spike traps as "pixels".

-- The video's resolution and frame rate are filled in by the Python script.
local videoData = {{
    width = {width},
    height = {height},
    fps = {fps},
    -- This 'data' table will hold a long string of numbers for each frame.
    -- Each number represents the color of one pixel in that frame.
    data = {{
{frame_data_lua}
    }}
}}

-- This table maps the number from the data string (e.g., "3") to a color index.
local pixelColorMap = {{
    ["0"] = 0, -- Black
    ["1"] = 1, -- Red
    ["2"] = 2, -- Yellow
    ["3"] = 3, -- Green
    ["4"] = 4, -- Blue
    ["5"] = 5  -- White
}}

-- This table maps our color index to the actual color names used by Chaoshead's spike traps.
-- We can choose which colors we want to actually create traps for.
-- In this case, we're creating traps for all colors except Black.
local chaosheadColorMap = {{
    [1] = "Red",
    [2] = "Yellow",
    [3] = "Green",
    [4] = "Blue",
    [5] = "White"
}}

-- =============================================================================
-- SETUP & CHANNELING
-- This section calculates how to wire everything together using the game's channel system.
-- Think of a channel like a wire or a radio frequency that can send an "on" or "off" signal.
-- =============================================================================

-- Core numbers used for calculations
local PIXELS_PER_FRAME = videoData.width * videoData.height
local NUM_FRAMES_IN_VIDEO = #videoData.data

local num_colors_with_relays_traps = 0
for _ in pairs(chaosheadColorMap) do num_colors_with_relays_traps = num_colors_with_relays_traps + 1 end

-- Channel Allocation Strategy: A master plan for our wiring.
-- We need to reserve blocks of channels for different purposes to avoid conflicts.
--
-- Block 1: Game Sequencer Triggers (Channels 0 to NUM_FRAMES_IN_VIDEO - 1)
-- These are the input channels. The game engine is expected to send a signal on
-- channel 0 for frame 1, channel 1 for frame 2, and so on.
local CH_OFFSET_SEQUENCER_TRIGGERS = 0

-- Block 2: Unique Frame Definition Triggers (Starts after Block 1)
-- This is a major optimization. If a video has repeating frames (e.g., a black screen),
-- we only build the logic for that frame once. This channel block gives each unique
-- frame pattern its own "master switch".
local CH_OFFSET_UNIQUE_FRAME_DEFINITION_TRIGGERS = CH_OFFSET_SEQUENCER_TRIGGERS + NUM_FRAMES_IN_VIDEO

-- Block 3: Pixel Activator Channels (Starts after Block 2)
-- These are the channels for the individual spike traps. There is one channel for
-- every single pixel, for every single color. For example: "Red trap at row 1, col 1",
-- "Blue trap at row 1, col 1", etc. This requires a huge number of channels.
local CH_OFFSET_PIXEL_ACTIVATORS = CH_OFFSET_UNIQUE_FRAME_DEFINITION_TRIGGERS + NUM_FRAMES_IN_VIDEO


-- Base coordinates for placing objects in the game world. We'll start at the top-left.
local bx = level.left
local by = level.top

-- Helper function to place a relay and configure it.
local function place_configured_relay(receive_channel, send_channel)
    local relay = level:placeRelay(bx, by)
    relay:setReceivingChannel(receive_channel)
    relay:setSendingChannel(send_channel)
    relay:setSwitchRequirements("Any Active")
    bx = bx + 1 -- Move one step to the right for the next object
    if bx > level.right then bx = level.left; by = by + 1 end -- If we hit the edge, go to the next line
end

-- Helper function to place a spike trap and configure it.
local function place_configured_spike_trap(receive_channel, trap_color)
    local trap = level:placeSpikeTrap(bx, by)
    trap:setReceivingChannel(receive_channel)
    trap:setColor(trap_color)
    trap:setSwitchRequirements("Any Active")
    bx = bx + 1 -- The grid layout is managed later in the trap placement section
end

-- =============================================================================
-- SECTION 1: Build the Logic (Placing Relays)
-- This section builds the complex web of relays that form the video player's brain.
-- It connects the game's timer (Sequencer Triggers) to the correct frame patterns.
-- =============================================================================
local uniqueFrames = {{}} -- A table to keep track of frame patterns we've already built.
local unique_frame_id_counter = 0 -- Counts how many unique frames we've found.

print("Starting relay generation...")

for frameIndex, frameDataString in ipairs(videoData.data) do
    -- This is the input channel from the game for the current frame number.
    local game_sequencer_trigger_channel = CH_OFFSET_SEQUENCER_TRIGGERS + (frameIndex - 1)

    -- Have we seen this exact pattern of pixels before?
    if not uniqueFrames[frameDataString] then
        -- NO: This is a new, unique frame. We need to build its logic from scratch.
        
        -- Assign a new "master switch" channel for this unique frame pattern.
        local unique_frame_definition_channel = CH_OFFSET_UNIQUE_FRAME_DEFINITION_TRIGGERS + unique_frame_id_counter
        uniqueFrames[frameDataString] = unique_frame_definition_channel
        
        -- Relay 1: Connect the game's timer to our new "master switch".
        -- When the game says "play frame X", this relay will flip our master switch.
        place_configured_relay(game_sequencer_trigger_channel, unique_frame_definition_channel)

        -- Now, create relays that connect the "master switch" to all the correct pixel traps.
        for pixel_index_in_frame = 0, PIXELS_PER_FRAME - 1 do
            local pixel_char = frameDataString:sub(pixel_index_in_frame + 1, pixel_index_in_frame + 1)
            local color_index = pixelColorMap[pixel_char]

            -- Only create relays for colors that actually have traps (e.g., not for black).
            if color_index and chaosheadColorMap[color_index] then
                
                -- We need to calculate the EXACT channel for this specific pixel of this specific color.
                -- First, get a predictable 0-indexed number for the color (e.g. Red=0, Yellow=1, etc.)
                local sorted_trap_color_keys = {{}}
                for k in pairs(chaosheadColorMap) do table.insert(sorted_trap_color_keys, k) end
                table.sort(sorted_trap_color_keys)
                local trap_color_block_index = 0
                for i, key in ipairs(sorted_trap_color_keys) do
                    if key == color_index then
                        trap_color_block_index = i - 1
                        break
                    end
                end

                -- The final channel is calculated by starting at the Pixel Activator block and then
                -- finding the specific offset for this pixel's position and color.
                local relative_pixel_channel_id = pixel_index_in_frame + (trap_color_block_index * PIXELS_PER_FRAME)
                local actual_pixel_activator_channel = CH_OFFSET_PIXEL_ACTIVATORS + relative_pixel_channel_id
                
                -- Relay 2..N: When the master switch is on, this relay turns on one specific trap.
                place_configured_relay(unique_frame_definition_channel, actual_pixel_activator_channel)
            end
        end
        unique_frame_id_counter = unique_frame_id_counter + 1
    else
        -- YES: We've seen this frame before. It's a duplicate.
        -- This is the optimization! We don't need to build anything new.
        
        -- Just get the "master switch" channel for the pattern we already built.
        local original_unique_frame_definition_channel = uniqueFrames[frameDataString]
        
        -- Relay: Connect the game's timer directly to that existing master switch.
        place_configured_relay(game_sequencer_trigger_channel, original_unique_frame_definition_channel)
    end
end
print("Finished placing relays. Found " .. unique_frame_id_counter .. " unique frames out of " .. NUM_FRAMES_IN_VIDEO .. " total frames.")


-- =============================================================================
-- SECTION 2: Build the Display (Placing Traps)
-- Now that the logic is built, this section places the physical spike traps
-- that will act as the pixels of our video screen.
-- =============================================================================
print("Placing spike traps...")
-- Reset placement coordinates for the trap grid.
bx = level.left
if unique_frame_id_counter > 0 or NUM_FRAMES_IN_VIDEO > 0 then
    by = by + 1 -- Move down one line to avoid placing traps on top of relays.
end

-- We will now create a massive grid of traps. A full set of traps (one for each color)
-- will be placed for every single pixel coordinate.
local total_pixel_channels_for_traps = num_colors_with_relays_traps * PIXELS_PER_FRAME

for i = 0, total_pixel_channels_for_traps - 1 do
    -- This is the receiving channel for the trap we're about to place.
    -- It must exactly match one of the `actual_pixel_activator_channel`s calculated in Section 1.
    local actual_trap_receive_channel = CH_OFFSET_PIXEL_ACTIVATORS + i

    -- From the channel number, we can work backwards to figure out this trap's color and position.
    local trap_color_block_index = math.floor(i / PIXELS_PER_FRAME)
    local pixel_index_in_frame_for_trap = i % PIXELS_PER_FRAME

    -- Get the trap color name (e.g., "Red", "Blue") from the color block index.
    local trap_color_name = nil
    local sorted_trap_color_keys_s2 = {{}}
    for k in pairs(chaosheadColorMap) do table.insert(sorted_trap_color_keys_s2, k) end
    table.sort(sorted_trap_color_keys_s2)
    local original_color_idx = sorted_trap_color_keys_s2[trap_color_block_index + 1]
    trap_color_name = chaosheadColorMap[original_color_idx]
    
    if trap_color_name then
        -- Place the trap with the correct channel and color.
        place_configured_spike_trap(actual_trap_receive_channel, trap_color_name)
        
        -- This logic arranges the traps into neat grids, one grid for each color.
        -- When a row of traps for one color is complete, it moves to the next line.
        if ((pixel_index_in_frame_for_trap + 1) % videoData.width) == 0 then
            bx = level.left
            by = by + 1
        end
    end
end
print("Finished placing traps.")
"""

def process_video_color_mapping_to_lua(video_filename, output_lua_filename, palette, labels,
                                     width, height, fps_target):
    """
    Orchestrates the process of reading the video, mapping colors, and generating the Lua script.

    Args:
        video_filename (str): The name of the low-res video file to process.
        output_lua_filename (str): The name for the final Lua script file.
        palette (np.array): The color palette to use for mapping.
        labels (np.array): The character labels for each palette color.
        width (int): The width of the video, needed for processing.
        height (int): The height of the video, needed for processing.
        fps_target (int): The target FPS, to be written into the Lua script.
    """
    video_path = os.path.join(os.getcwd(), video_filename)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_filename}")
        return

    frame_data_strings = []
    frame_count = 0
    
    # Get some info from the processed video file.
    actual_fps_of_processed_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_processed_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing frames from '{video_filename}'...")
    start_time = time.time()
    
    # Loop through every frame of the small video.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Stop if there are no more frames.
        
        frame_count += 1
        
        # This is a safety check. If a frame isn't the expected size for some reason,
        # it will be resized on the fly. This shouldn't happen if the first step worked correctly.
        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"Warning: Frame {frame_count} has unexpected dimensions. Resizing.")
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        # Convert the frame's pixels to our limited color palette.
        closest_indices = map_colors_to_palette_vectorized(frame, palette)
        # Get the character labels for the resulting color indices.
        frame_labels_chars = labels[closest_indices]
        # Join the characters into a single long string representing the entire frame.
        frame_string = "".join(frame_labels_chars)
        frame_data_strings.append(frame_string)
        
    cap.release()
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds.")

    if not frame_data_strings:
        print("No frames were processed from the video.")
        return

    # Format the list of frame strings into a Lua-compatible table format.
    lua_formatted_frames = []
    for i, frame_str in enumerate(frame_data_strings):
        line = f'        "{frame_str}"'
        # Add a comma after each line except the very last one.
        if i < len(frame_data_strings) - 1:
            line += ','
        lua_formatted_frames.append(line)
    frame_data_lua_string = "\n".join(lua_formatted_frames)

    # Insert all our data (width, height, fps, frames) into the Lua template.
    lua_script_content = LUA_SCRIPT_TEMPLATE.format(
        width=width,
        height=height,
        fps=fps_target,
        frame_data_lua=frame_data_lua_string
    )

    # Write the final, complete Lua script to a file.
    output_lua_path = os.path.join(os.getcwd(), output_lua_filename)
    with open(output_lua_path, 'w') as file:
        file.write(lua_script_content)
    print(f"Success! Lua script generated: {output_lua_path}")

# --- SCRIPT EXECUTION ---
# This is the main part of the script that runs when you execute the file.
if __name__ == "__main__":
    
    # --- Configuration ---
    # You can change these values to customize the output.
    original_video_filename = "video.mp4"
    low_res_video_filename = "video_low_res.mp4"
    color_mapped_output_lua_filename = "chaoshead_video_data.lua"
    
    # The dimensions of the in-game display.
    target_width = 11
    target_height = 8
    # The frames-per-second of the in-game playback.
    target_fps = 1 

    # If 'video.mp4' doesn't exist, this block creates a sample video for testing.
    # This ensures the script can run even without a user-provided video.
    if not os.path.exists(original_video_filename):
        print(f"'{original_video_filename}' not found. Creating a dummy video for testing.")
        dummy_width, dummy_height = 64, 48
        dummy_source_fps = 10
        dummy_duration_seconds = 2.5
        dummy_total_frames = int(dummy_source_fps * dummy_duration_seconds)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_dummy = cv2.VideoWriter(original_video_filename, fourcc, dummy_source_fps, (dummy_width, dummy_height))
        
        if not out_dummy.isOpened():
            print(f"Error: Failed to create the dummy video file. Check your OpenCV/ffmpeg setup.")
        else:
            # Create a few simple, colored patterns for the dummy video
            frame_pattern_A = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
            frame_pattern_A[:, :dummy_width//2, 2] = 255  # Red left half

            frame_pattern_B = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
            frame_pattern_B[dummy_height//2:, :, 1] = 255  # Green bottom half
            
            frame_pattern_C = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
            cv2.circle(frame_pattern_C, (dummy_width//2, dummy_height//2), dummy_height//4, (0,0,255), -1) # Red circle

            # Create a sequence of patterns with some repetition to test the "unique frame" logic
            sequence_patterns = [frame_pattern_A, frame_pattern_B, frame_pattern_A, frame_pattern_C, 
                                 frame_pattern_B, frame_pattern_A, frame_pattern_C, frame_pattern_C]
            
            for i in range(dummy_total_frames):
                current_pattern = sequence_patterns[i % len(sequence_patterns)]
                out_dummy.write(current_pattern)
            out_dummy.release()
            print(f"Dummy video created successfully.")

    # --- Main Process ---
    # 1. Convert the original video to the low-resolution version.
    if convert_and_resize_video(original_video_filename, low_res_video_filename, target_width, target_height, target_fps):
        # 2. If conversion is successful, process the low-res video and generate the Lua script.
        process_video_color_mapping_to_lua(low_res_video_filename, color_mapped_output_lua_filename, palette_bgr, palette_labels, target_width, target_height, target_fps)
