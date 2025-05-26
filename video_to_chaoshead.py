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
        return False

    print(f"Converting and resizing video: {input_filename} to {output_path} ({target_width}x{target_height}, {target_fps} fps)...")

    clip = None
    resized_clip = None
    final_clip = None
    try:
        clip = VideoFileClip(input_path)
        resized_clip = clip.resize(newsize=(target_width, target_height))
        final_clip = resized_clip.set_fps(target_fps)
        final_clip.write_videofile(output_path, codec='libx264', logger=None)
        print("Video conversion and resizing complete.")
        return True
    except Exception as e:
        print(f"Error during video conversion: {e}")
        return False
    finally:
        if clip:
            clip.close()
        if resized_clip:
            resized_clip.close()
        if final_clip:
            final_clip.close()

# --- Color Mapping Palette and Labels ---
palette_bgr = np.array([
    [0, 0, 0],      # 0: Black
    [0, 0, 255],    # 1: Red
    [0, 255, 255],  # 2: Yellow
    [0, 255, 0],    # 3: Green
    [255, 0, 0],    # 4: Blue
    [255, 255, 255] # 5: White
], dtype=np.uint8)

palette_labels = np.array(['0', '1', '2', '3', '4', '5'])

def map_colors_to_palette_vectorized(frame, palette):
    pixels = frame.reshape(-1, 3).astype(np.float32)
    palette_float = palette.astype(np.float32)
    distances_sq = np.sum((pixels[:, np.newaxis, :] - palette_float[np.newaxis, :, :])**2, axis=2)
    indices = np.argmin(distances_sq, axis=1)
    return indices

# --- Lua Script Template ---
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

local pixelColorMap = {{
    ["0"] = 0,
    ["1"] = 1,
    ["2"] = 2,
    ["3"] = 3,
    ["4"] = 4,
    ["5"] = 5
}}

local chaosheadColorMap = {{
    [1] = "Red",
    [2] = "Yellow",
    [3] = "Green",
    [4] = "Blue",
    [5] = "White"
}}

local channels_per_color = videoData.width * videoData.height
local num_colors_in_palette = 0
for _ in pairs(pixelColorMap) do num_colors_in_palette = num_colors_in_palette + 1 end
local num_colors_with_relays_traps = num_colors_in_palette - 1
local frame_trigger_start_channel = num_colors_with_relays_traps * channels_per_color

-- Base placement coords
local bx = level.left
local by = level.top

local function place_configured_relay(receive_channel, send_channel)
    local b = level:placeRelay(bx, by)
    b:setReceivingChannel(receive_channel)
    b:setSendingChannel(send_channel)
    b:setSwitchRequirements("Any Active")
    bx = bx + 1
    if bx > level.right then bx = level.left; by = by + 1 end
    return b
end

local function place_configured_spike_trap(receive_channel, trap_color)
    local trap = level:placeSpikeTrap(bx, by)
    trap:setReceivingChannel(receive_channel)
    trap:setColor(trap_color)
    trap:setSwitchRequirements("Any Active")
    bx = bx + 1
    -- Note: newline logic moved to loop below for consistent width-based wrapping
    return trap
end

-- SECTION 1: Relays
local uniqueFrames = {{}}
local unique_frame_count = 0
local current_frame_trigger_channel = frame_trigger_start_channel
print("Starting relay generation...")
for frameIndexWithinData, frameData in ipairs(videoData.data) do
    if not uniqueFrames[frameData] then
        unique_frame_count = unique_frame_count + 1
        local frame_receive_channel = current_frame_trigger_channel
        uniqueFrames[frameData] = frame_receive_channel
        local pixel_index_in_frame = 0
        for pixel_char in frameData:gmatch(".") do
            local color_index = pixelColorMap[pixel_char]
            if color_index and color_index ~= 0 then
                local shifted_color_index = color_index - 1
                local pixel_send_channel = pixel_index_in_frame + shifted_color_index * channels_per_color
                place_configured_relay(frame_receive_channel, pixel_send_channel)
            end
            pixel_index_in_frame = pixel_index_in_frame + 1
        end
        current_frame_trigger_channel = current_frame_trigger_channel + 1
    else
        local original_frame_receive_channel = uniqueFrames[frameData]
        local implicit_frame_data_trigger_channel = frameIndexWithinData - 1
        place_configured_relay(implicit_frame_data_trigger_channel, original_frame_receive_channel)
    end
end
print("Finished placing relays.")

-- SECTION 2: Traps
print("Placing spike traps...")
-- Start traps on a new line
bx = level.left
by = by + 1
local total_pixel_channels_with_traps = num_colors_with_relays_traps * channels_per_color
for pixel_send_channel = 0, total_pixel_channels_with_traps - 1 do
    local shifted_color_index = math.floor(pixel_send_channel / channels_per_color)
    local color_index = shifted_color_index + 1
    local trap_color_name = chaosheadColorMap[color_index]
    if trap_color_name then
        place_configured_spike_trap(pixel_send_channel, trap_color_name)
        -- Wrap after each row of width traps
        if ((pixel_send_channel + 1) % videoData.width) == 0 then
            bx = level.left
            by = by + 1
        end
    end
end
print("Finished placing traps.")"""

# Function to process video frames and export color-mapped data to a Lua script file
def process_video_color_mapping_to_lua(video_filename, output_lua_filename, palette, labels,
                                     width, height, fps):
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
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    print(f"Processing frames from video: {video_filename} (Estimated total frames: {total_frames_estimate})")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        closest_indices = map_colors_to_palette_vectorized(frame, palette)
        frame_labels_chars = labels[closest_indices]
        frame_string = "".join(frame_labels_chars)
        frame_data_strings.append(frame_string)
    cap.release()
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f}s.")

    if not frame_data_strings:
        print("No frames processed.")
        return

    lua_formatted_frames = []
    for i, frame_str in enumerate(frame_data_strings):
        line = f'        "{frame_str}"'
        if i < len(frame_data_strings) - 1:
            line += ','
        lua_formatted_frames.append(line)
    frame_data_lua_string = "\n".join(lua_formatted_frames)

    lua_script_content = LUA_SCRIPT_TEMPLATE.format(
        width=width,
        height=height,
        fps=fps,
        frame_data_lua=frame_data_lua_string
    )

    with open(output_lua_filename, 'w') as file:
        file.write(lua_script_content)

if __name__ == "__main__":
    original_video_filename = "video.mp4"
    low_res_video_filename = "video_low_res.mp4"
    color_mapped_output_lua_filename = "chaoshead_video_data.lua"
    target_width = 12
    target_height = 9
    target_fps = 2
    if convert_and_resize_video(original_video_filename, low_res_video_filename, target_width, target_height, target_fps):
        process_video_color_mapping_to_lua(low_res_video_filename, color_mapped_output_lua_filename, palette_bgr, palette_labels, target_width, target_height, target_fps)
