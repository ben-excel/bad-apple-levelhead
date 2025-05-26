import cv2
import os
import numpy as np
import time
from moviepy.editor import VideoFileClip

# --- Video Conversion & Resizing (using moviepy) ---
def convert_and_resize_video(input_filename, output_filename, target_width, target_height, target_fps):
    """
    Resizes a video and sets its frame rate using moviepy.

    Args:
        input_filename (str): The path to the input video file.
        output_filename (str): The desired path for the output video file.
        target_width (int): The desired width of the output video.
        target_height (int): The desired height of the output video.
        target_fps (int): The desired frame rate of the output video.
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    current_directory = os.getcwd()
    input_path = os.path.join(current_directory, input_filename)
    output_path = os.path.join(current_directory, output_filename)

    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at {input_path}")
        return False # Indicate failure

    print(f"Converting and resizing video: {input_filename} to {output_path} ({target_width}x{target_height}, {target_fps} fps)...")

    clip = None
    resized_clip = None
    final_clip = None
    try:
        # Load the video
        clip = VideoFileClip(input_path)

        # Resize the video
        resized_clip = clip.resize(newsize=(target_width, target_height))

        # Set the frame rate
        final_clip = resized_clip.set_fps(target_fps)

        # Save the modified video
        final_clip.write_videofile(output_path, codec='libx264', logger=None)

        print("Video conversion and resizing complete.")
        return True # Indicate success

    except Exception as e:
        print(f"Error during video conversion: {e}")
        return False # Indicate failure
    finally:
        # Ensure clips are closed to release resources, regardless of success or failure
        if clip:
            clip.close()
        if resized_clip:
            resized_clip.close()
        if final_clip:
            final_clip.close()

# --- Color Mapping Palette and Labels ---
# Defines the color palette used for simplifying video frames.
# Colors are in BGR format (OpenCV default).
palette_bgr = np.array([
    [0, 0, 0],      # 0: Black
    [0, 0, 255],    # 1: Red
    [0, 255, 255],  # 2: Yellow
    [0, 255, 0],    # 3: Green
    [255, 0, 0],    # 4: Blue
    [255, 255, 255] # 5: White
], dtype=np.uint8)

# Corresponding character labels for each color in the palette.
palette_labels = np.array(['0', '1', '2', '3', '4', '5'])

# --- Color Mapping Logic (using cv2 and numpy) ---
def map_colors_to_palette_vectorized(frame, palette):
    """
    Maps each pixel in a frame to the closest color in the given palette.

    This function uses a vectorized approach for efficiency by calculating
    squared Euclidean distances between all pixels and all palette colors
    simultaneously.

    Args:
        frame (np.ndarray): The input video frame (BGR format).
        palette (np.ndarray): The color palette (BGR format).

    Returns:
        np.ndarray: An array of indices representing the closest palette color
                    for each pixel in the original frame's flattened view.
    """
    # Reshape frame to a list of pixels (N, 3) and convert to float for distance calculation
    pixels = frame.reshape(-1, 3).astype(np.float32)
    palette_float = palette.astype(np.float32)

    # Calculate squared Euclidean distances: (pixels[:, np.newaxis, :] - palette_float[np.newaxis, :, :])
    # creates a difference tensor. Squaring and summing along axis 2 gives squared distances.
    # Shape of pixels[:, np.newaxis, :]: (num_pixels, 1, 3)
    # Shape of palette_float[np.newaxis, :, :]: (1, num_palette_colors, 3)
    # Resulting shape of distances_sq: (num_pixels, num_palette_colors)
    distances_sq = np.sum((pixels[:, np.newaxis, :] - palette_float[np.newaxis, :, :])**2, axis=2)

    # Find the index of the palette color with the minimum distance for each pixel
    indices = np.argmin(distances_sq, axis=1)
    return indices

# --- Lua Script Template ---
# This template defines the structure of the Lua script generated to control
# "Chaoshead" game elements based on the processed video data.
LUA_SCRIPT_TEMPLATE = """-- Lua script to process video data, generate relays, and place pixel traps in Chaoshead

-- IMPORTANT: Update width and height to match the Python script's output resolution!
local videoData = {{
    width = {width},
    height = {height},
    data = {{
{frame_data_lua}
    }},
    fps = {fps}
}}

-- Map pixel characters ('0' through '5') to their integer indices (0 through 5).
-- This MUST match the order in your Python script's palette_labels array.
local pixelColorMap = {{
    ["0"] = 0, -- Black (Index 0) - Relays and Traps for this color are SKIPPED
    ["1"] = 1, -- Red
    ["2"] = 2, -- Yellow
    ["3"] = 3, -- Green
    ["4"] = 4, -- Blue
    ["5"] = 5  -- White
}}

-- Map color indices (for colors that place traps) to Chaoshead color names.
-- Index 0 (Black) is intentionally excluded.
-- The indices must match the values in pixelColorMap that are > 0.
local chaosheadColorMap = {{
    [1] = "Red",
    [2] = "Yellow",
    [3] = "Green",
    [4] = "Blue",
    [5] = "White"
}}

-- Calculate constants based on video data and palette
local channels_per_color = videoData.width * videoData.height -- Total pixels per frame
local num_colors_in_palette = 0
for _ in pairs(pixelColorMap) do
    num_colors_in_palette = num_colors_in_palette + 1
end

-- Determine how many colors will actually place relays/traps (those with index > 0)
local num_colors_with_relays_traps = num_colors_in_palette - 1 -- Assuming index 0 (Black) is the only one skipped

-- Calculate the starting channel for frame-triggering signals.
-- These channels trigger unique frame patterns and start AFTER all pixel-specific channels.
-- Pixel-specific channels range from 0 up to (num_colors_with_relays_traps * channels_per_color - 1).
local frame_trigger_start_channel = num_colors_with_relays_traps * channels_per_color

-- Base position for placing items (defined by Chaoshead 'level' object).
-- Assumes 'level.left', 'level.top', and 'level.right' are accessible in the Chaoshead environment.
local bx = level.left
local by = level.top

-- Helper function to place and configure a relay
local function place_configured_relay(receive_channel, send_channel)
    local b = level:placeRelay(bx, by)
    b:setReceivingChannel(receive_channel)
    b:setSendingChannel(send_channel)
    b:setSwitchRequirements("Any Active") -- Adjust switch requirements as needed

    -- Advance placement position for the next item
    bx = bx + 1
    if bx > level.right then
        bx = level.left -- Reset X to start of next row
        by = by + 1
    end
    return b -- Return the placed item object if needed
end

-- Helper function to place and configure a spike trap for a pixel channel
local function place_configured_spike_trap(receive_channel, trap_color)
    local trap = level:placeSpikeTrap(bx, by)
    trap:setReceivingChannel(receive_channel)
    trap:setColor(trap_color)
    trap:setSwitchRequirements("Any Active") -- Adjust switch requirements as needed

    -- Advance placement position for the next item
    bx = bx + 1
    if bx > level.right then
        bx = level.left -- Reset X to start of next row
        by = by + 1
    end
    return trap -- Return the placed item object if needed
end


-- Data structures to track unique frames and channel assignment
local uniqueFrames = {{}} -- Maps frame data string -> its unique frame trigger channel
local unique_frame_count = 0 -- Counter for unique frame patterns found

-- Channel assignment for unique frame triggers
local current_frame_trigger_channel = frame_trigger_start_channel

print("Starting relay generation based on video data...")
print("Video dimensions: " .. videoData.width .. "x" .. videoData.height)
print("Total pixels per frame: " .. channels_per_color)
print("Number of colors in palette: " .. num_colors_in_palette)
print("Number of colors that place relays/traps: " .. num_colors_with_relays_traps)
print("Pixel signal channels range: 0 to " .. (frame_trigger_start_channel - 1))
print("Frame trigger channels start from: " .. frame_trigger_start_channel)

-- --- SECTION 1: Place Relays ---
-- Process each frame string in the video data to place relays for unique frame patterns.
-- This optimizes by creating one set of pixel-activating relays per unique frame,
-- and then simpler relays to trigger these sets for duplicate frames.
for frameIndexWithinData, frameData in ipairs(videoData.data) do
    -- Check if this frame's data string has been encountered before
    if not uniqueFrames[frameData] then
        -- This is a new, unique frame pattern
        unique_frame_count = unique_frame_count + 1

        -- Assign a unique frame trigger channel for this pattern
        local frame_receive_channel = current_frame_trigger_channel
        uniqueFrames[frameData] = frame_receive_channel

        print("Processing unique frame pattern #" .. unique_frame_count .. " (Frame index in data: " .. frameIndexWithinData .. "). Assigning frame trigger channel: " .. frame_receive_channel)

        -- Iterate through each pixel character in the frame string
        local pixel_index_in_frame = 0 -- 0-based index for the pixel within the frame
        for pixel_char in frameData:gmatch(".") do
            local color_index = pixelColorMap[pixel_char]

            if color_index ~= nil then
                -- Only place a relay if the color index is NOT 0 (Black)
                if color_index ~= 0 then
                    -- Calculate the sending channel for this specific pixel's color and position.
                    -- (color_index - 1) shifts non-black colors to start from index 0 for channel block calculation.
                    local shifted_color_index = color_index - 1
                    local pixel_send_channel = pixel_index_in_frame + shifted_color_index * channels_per_color

                    -- Place a relay: receives the frame's unique trigger, sends to the pixel-specific channel.
                    place_configured_relay(frame_receive_channel, pixel_send_channel)
                    -- print(string.format("  Relay: Pixel %d ('%s', color index %d -> shifted %d) -> (%d -> %d)",
                    --                       pixel_index_in_frame, pixel_char, color_index, shifted_color_index,
                    --                       frame_receive_channel, pixel_send_channel))
                end
            else
                 print("Warning: Encountered unexpected pixel character '" .. pixel_char .. "' at pixel index " .. pixel_index_in_frame .. " in frame " .. frameIndexWithinData .. ". Skipping item for this pixel.")
            end
            pixel_index_in_frame = pixel_index_in_frame + 1
        end

        -- Increment the frame trigger channel for the NEXT unique frame pattern
        current_frame_trigger_channel = current_frame_trigger_channel + 1
    else
        -- This frame pattern has been seen before (it's a duplicate)
        local original_frame_receive_channel = uniqueFrames[frameData]
        print("Processing duplicate frame pattern (Frame index in data: " .. frameIndexWithinData .. "). Links to frame trigger channel: " .. original_frame_receive_channel)

        -- Place a single relay:
        -- It receives the trigger for THIS frame's position in the sequence (implicit_frame_data_trigger_channel)
        -- and sends it to the channel that activates relays for the original unique frame pattern.
        -- Assumes frame `i` in videoData.data is triggered by an external channel `i - 1`.
        local implicit_frame_data_trigger_channel = frameIndexWithinData - 1
        place_configured_relay(implicit_frame_data_trigger_channel, original_frame_receive_channel)
        -- print(string.format("  Relay: Duplicate frame %d -> (%d -> %d)",
        --                       frameIndexWithinData, implicit_frame_data_trigger_channel,
        --                       original_frame_receive_channel))
    end
    -- bx, by are updated inside place_configured_relay
end

print("Finished placing relays. Continuing placement for spike traps...")

-- --- SECTION 2: Place Spike Traps for each possible pixel channel ---
-- These traps listen on the channels that the pixel relays (from unique frames) send on.
-- They are colored according to which color block their channel falls into.
-- Total traps = (width * height * number_of_trap_placing_colors).
local total_pixel_channels_with_traps = num_colors_with_relays_traps * channels_per_color
print("Placing " .. total_pixel_channels_with_traps .. " spike traps...")

for pixel_send_channel = 0, total_pixel_channels_with_traps - 1 do
    -- Determine the shifted color index and pixel index from the linear channel number
    local shifted_color_index = math.floor(pixel_send_channel / channels_per_color)
    local pixel_index_in_frame = pixel_send_channel % channels_per_color

    -- Calculate the actual color index (add 1 back, as index 0 was for black/skipped)
    local color_index = shifted_color_index + 1

    -- Get the corresponding Chaoshead color name for the trap
    local trap_color_name = chaosheadColorMap[color_index]

    if trap_color_name ~= nil then
        place_configured_spike_trap(pixel_send_channel, trap_color_name)
        -- print(string.format("  Trap: Channel %d (Pixel %d, Color Index %d -> '%s')",
        --                       pixel_send_channel, pixel_index_in_frame, color_index, trap_color_name))
    else
        print("Error: Could not determine color for channel " .. pixel_send_channel .. ". Skipping trap.")
    end
    -- bx, by are updated inside place_configured_spike_trap
end

print("Finished generating items based on video data.")
print("Total frames processed from data: " .. #videoData.data)
print("Total unique frame patterns found: " .. unique_frame_count)
print("Total pixel relay channels used (0 to " .. (frame_trigger_start_channel - 1) .. "): " .. frame_trigger_start_channel)
print("Total unique frame trigger channels used (" .. frame_trigger_start_channel .. " to " .. (current_frame_trigger_channel - 1) .. "): " .. unique_frame_count)
-- Note: To get exact counts of relays/traps placed, counters would need to be incremented
-- within the place_configured_relay/trap functions during their execution in Chaoshead.
-- Estimated total relays placed = unique_frame_count (for duplicate frame links) + Sum(non-black pixels in each unique frame pattern)
-- Estimated total traps placed = num_colors_with_relays_traps * channels_per_color

-- END OF LUA SCRIPT
"""


# Function to process video frames and export color-mapped data to a Lua script file
def process_video_color_mapping_to_lua(video_filename, output_lua_filename, palette, labels,
                                     width, height, fps):
    """
    Reads frames from a video file, maps pixel colors to a defined palette,
    and embeds this data into a Lua script for use in Chaoshead.

    Args:
        video_filename (str): Path to the (resized) input video file.
        output_lua_filename (str): Path for the output Lua script file.
        palette (np.ndarray): The BGR color palette.
        labels (np.ndarray): Character labels for the palette colors.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        fps (int): Frame rate of the video.
    """
    video_path = os.path.join(os.getcwd(), video_filename)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_filename}")
        return

    frame_data_strings = [] # Stores string representation of each processed frame
    frame_count = 0
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    print(f"Processing frames from video: {video_filename} (Estimated total frames: {total_frames_estimate if total_frames_estimate > 0 else 'N/A'})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video or error

        frame_count += 1
        # Progress reporting
        if total_frames_estimate > 0 and frame_count % max(1, total_frames_estimate // 20) == 0: # Approx every 5%
              print(f"Processing frame {frame_count}/{total_frames_estimate}...")
        elif frame_count % 50 == 0: # Fallback for videos without frame count or very short videos
              print(f"Processing frame {frame_count}...")

        # Map frame pixels to the palette and convert to a string of labels
        closest_indices = map_colors_to_palette_vectorized(frame, palette)
        frame_labels_chars = labels[closest_indices]
        frame_string = "".join(frame_labels_chars)
        frame_data_strings.append(frame_string)

    cap.release()
    end_time = time.time()
    print(f"Finished processing {frame_count} frames in {end_time - start_time:.2f} seconds.")

    if not frame_data_strings:
        print("No frames processed or video was empty.")
        return

    # Format frame data for the Lua table structure
    lua_formatted_frames = []
    for i, frame_str in enumerate(frame_data_strings):
        line = f'        "{frame_str}"' # Indentation for Lua table entry
        if i < len(frame_data_strings) - 1:
            line += ',' # Add comma for all but the last frame string
        lua_formatted_frames.append(line)
    
    frame_data_lua_string = "\n".join(lua_formatted_frames)

    # Populate the Lua script template with video data
    lua_script_content = LUA_SCRIPT_TEMPLATE.format(
        width=width,
        height=height,
        fps=fps,
        frame_data_lua=frame_data_lua_string
    )

    # Write the generated Lua script to a file
    output_lua_path = os.path.join(os.getcwd(), output_lua_filename)
    print(f"Writing Lua script to {output_lua_path}...")
    try:
        with open(output_lua_path, 'w') as file:
            file.write(lua_script_content)
        print(f"Lua script written to {output_lua_path}")
    except IOError as e:
        print(f"Error writing Lua script to file {output_lua_path}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    original_video_filename = "video.mp4"       # Source video file
    low_res_video_filename = "video_low_res.mp4" # Intermediate resized video
    color_mapped_output_lua_filename = "chaoshead_video_data.lua" # Final Lua script output

    # Target parameters for video resizing and processing
    target_width = 12
    target_height = 9
    target_fps = 3

    print("--- Step 1: Video Conversion and Resizing ---")
    conversion_successful = convert_and_resize_video(
        original_video_filename,
        low_res_video_filename,
        target_width,
        target_height,
        target_fps
    )

    if conversion_successful:
        print("\n--- Step 2: Color Mapping and Lua Script Generation ---")
        process_video_color_mapping_to_lua(
            low_res_video_filename,
            color_mapped_output_lua_filename,
            palette_bgr,
            palette_labels,
            target_width,
            target_height,
            target_fps
        )
    else:
        print("\nSkipping color mapping and Lua script generation due to video conversion/resizing failure.")

    print("\n--- Script Finished ---")
