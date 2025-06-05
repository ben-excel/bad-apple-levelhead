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
        # Use logger=None to suppress moviepy's default console output during write_videofile
        final_clip.write_videofile(output_path, codec='libx264', logger=None, threads=4, preset='ultrafast')
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
    [28, 23, 24],      # 0: Black
    [53, 37, 253],    # 1: Red
    [0, 210, 253],  # 2: Yellow
    [98, 166, 0],    # 3: Green
    [215, 27, 37],    # 4: Blue
    [204, 204, 204] # 5: White
], dtype=np.uint8)

palette_labels = np.array(['0', '1', '2', '3', '4', '5'])

def map_colors_to_palette_vectorized(frame, palette):
    pixels = frame.reshape(-1, 3).astype(np.float32)
    palette_float = palette.astype(np.float32)
    # Using squared Euclidean distance for efficiency (argmin is the same)
    distances_sq = np.sum((pixels[:, np.newaxis, :] - palette_float[np.newaxis, :, :])**2, axis=2)
    indices = np.argmin(distances_sq, axis=1)
    return indices

# --- Lua Script Template (Incorporating detailed pixel channel explanation) ---
LUA_SCRIPT_TEMPLATE = """-- Lua script to process video data, generate relays, and place pixel traps in Chaoshead

-- Video resolution (width, height) and FPS are automatically set by the Python script.
local videoData = {{
    width = {width},
    height = {height},
    data = {{
{frame_data_lua}
    }},
    fps = {fps}
}}

local pixelColorMap = {{
    ["0"] = 0, -- Black
    ["1"] = 1, -- Red
    ["2"] = 2, -- Yellow
    ["3"] = 3, -- Green
    ["4"] = 4, -- Blue
    ["5"] = 5  -- White
}}

-- Chaoshead color names for traps. Green (index 3) is excluded. Black (index 0) is implicitly excluded.
local chaosheadColorMap = {{
    [1] = "Red",    -- Corresponds to pixelColorMap index 1
    [2] = "Yellow", -- Corresponds to pixelColorMap index 2
    [4] = "Blue",   -- Corresponds to pixelColorMap index 4
    [5] = "White"   -- Corresponds to pixelColorMap index 5
}}

-- Core constants for channel management
local PIXELS_PER_FRAME = videoData.width * videoData.height
local NUM_FRAMES_IN_VIDEO = #videoData.data

local num_colors_in_palette = 0
for _ in pairs(pixelColorMap) do num_colors_in_palette = num_colors_in_palette + 1 end
-- num_colors_with_relays_traps is the count of colors that will have associated relays/traps
-- (e.g., excluding black and green, so 6 - 2 = 4 colors if all others are in chaosheadColorMap)
local num_colors_with_relays_traps = 0
for color_idx, _ in pairs(chaosheadColorMap) do
    if pixelColorMap[tostring(color_idx)] then -- Check if this color index is in our main palette
         num_colors_with_relays_traps = num_colors_with_relays_traps + 1
    end
end


-- Channel Allocation Strategy:
-- 1. Game Sequencer Triggers (External Input):
--    Channels the game engine uses to signal which video frame to play.
--    Range: 0 to NUM_FRAMES_IN_VIDEO - 1.
local CH_OFFSET_SEQUENCER_TRIGGERS = 0

-- 2. Unique Frame Definition Triggers (Internal):
--    Channels that activate the specific set of relays for a unique frame's pixel pattern.
--    These start immediately after the sequencer trigger channels.
--    Range (potential, if all frames are unique): NUM_FRAMES_IN_VIDEO to (NUM_FRAMES_IN_VIDEO + NUM_FRAMES_IN_VIDEO - 1)
local CH_OFFSET_UNIQUE_FRAME_DEFINITION_TRIGGERS = CH_OFFSET_SEQUENCER_TRIGGERS + NUM_FRAMES_IN_VIDEO

-- 3. Pixel Activator Channels (Internal, for Traps):
--    Channels that individual traps listen to.
--    These start after all potential unique frame definition channels.
--    (Worst case for unique frames is NUM_FRAMES_IN_VIDEO unique frames).
--    The block of pixel activator channels has a size of: num_colors_with_relays_traps * PIXELS_PER_FRAME
local CH_OFFSET_PIXEL_ACTIVATORS = CH_OFFSET_UNIQUE_FRAME_DEFINITION_TRIGGERS + NUM_FRAMES_IN_VIDEO

-- Base placement coords for game objects
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
    -- Grid layout for traps is handled in SECTION 2
    return trap
end

-- SECTION 1: Relays
-- Purpose: For each frame in the video:
--   1. Link its game sequencer trigger channel to a unique frame definition channel.
--   2. If the frame's content is new, create relays from its unique frame definition channel
--      to the appropriate pixel activator channels based on its pixel data.
local uniqueFrames = {{}} -- Maps frameData string to its unique_frame_definition_channel
local unique_frame_id_counter = 0 -- 0-indexed ID for unique frames encountered

print("Starting relay generation...")
print("Channel Offsets & Info:")
print("  Game Sequencer Triggers start at: " .. CH_OFFSET_SEQUENCER_TRIGGERS .. " (Range: 0 to " .. (NUM_FRAMES_IN_VIDEO - 1) .. ")")
print("  Unique Frame Definitions start at: " .. CH_OFFSET_UNIQUE_FRAME_DEFINITION_TRIGGERS .. " (Max needed: " .. unique_frame_id_counter .. " up to " .. NUM_FRAMES_IN_VIDEO .. ")")
print("  Pixel Activators start at: " .. CH_OFFSET_PIXEL_ACTIVATORS)
print("  Pixels per frame: " .. PIXELS_PER_FRAME)
print("  Colors with traps: " .. num_colors_with_relays_traps)

for frameIndexWithinData, frameData in ipairs(videoData.data) do
    local game_sequencer_trigger_channel = CH_OFFSET_SEQUENCER_TRIGGERS + (frameIndexWithinData - 1)

    if not uniqueFrames[frameData] then
        -- This is a new, unique frame content
        local unique_frame_definition_channel = CH_OFFSET_UNIQUE_FRAME_DEFINITION_TRIGGERS + unique_frame_id_counter
        uniqueFrames[frameData] = unique_frame_definition_channel
        
        -- Relay 1: Links game's time-based trigger for this frame to its unique content definition.
        -- e.g., Game triggers channel 0 (for 1st frame) -> Activates unique_frame_definition_channel X
        place_configured_relay(game_sequencer_trigger_channel, unique_frame_definition_channel)

        -- For this unique frame definition, set up relays to activate individual pixel channels.
        local pixel_index_in_frame = 0 -- 0 to (PIXELS_PER_FRAME - 1)
        for pixel_char in frameData:gmatch(".") do
            local color_index_from_map = pixelColorMap[pixel_char] -- e.g., 0 for Black, 1 for Red

            -- Only create relays for colors that have traps (defined in chaosheadColorMap)
            if color_index_from_map and chaosheadColorMap[color_index_from_map] then
                -- To map chaosheadColorMap keys (1,2,4,5) to a compact 0-indexed sequence for channel blocks:
                local trap_color_block_index = 0
                local temp_idx = 0
                for ch_map_idx, _ in pairs(chaosheadColorMap) do -- Iterating pairs might not be ordered, use sorted keys for consistency
                    if ch_map_idx == color_index_from_map then
                        trap_color_block_index = temp_idx
                        break
                    end
                    temp_idx = temp_idx + 1
                end
                -- A more robust way to get trap_color_block_index if chaosheadColorMap keys are sparse:
                local sorted_trap_color_keys = {{}}
                for k in pairs(chaosheadColorMap) do table.insert(sorted_trap_color_keys, k) end
                table.sort(sorted_trap_color_keys)
                for i, key in ipairs(sorted_trap_color_keys) do
                    if key == color_index_from_map then
                        trap_color_block_index = i - 1 -- 0-indexed block for this color
                        break
                    end
                end

                -- relative_pixel_channel_id: This ID is unique for a pixel of a specific color at a specific location.
                -- It's 0-indexed within the entire conceptual block of "all pixels of all trappable colors".
                -- Formula: (pixel's position within the frame) + (color's block offset)
                -- Color's block offset: trap_color_block_index * PIXELS_PER_FRAME
                local relative_pixel_channel_id = pixel_index_in_frame + (trap_color_block_index * PIXELS_PER_FRAME)
                
                -- actual_pixel_activator_channel: The final global channel number this pixel (of this color, at this position) will use.
                local actual_pixel_activator_channel = CH_OFFSET_PIXEL_ACTIVATORS + relative_pixel_channel_id
                
                -- Relay 2..N: When unique_frame_definition_channel X is active, activate the specific pixel channel.
                place_configured_relay(unique_frame_definition_channel, actual_pixel_activator_channel)
            end
            pixel_index_in_frame = pixel_index_in_frame + 1
        end
        unique_frame_id_counter = unique_frame_id_counter + 1
    else
        -- This frame's content is a duplicate of a previously defined unique frame
        local original_unique_frame_definition_channel = uniqueFrames[frameData]
        
        -- Relay: Links game's time-based trigger directly to the existing unique content definition.
        place_configured_relay(game_sequencer_trigger_channel, original_unique_frame_definition_channel)
    end
end
print("Finished placing relays. Total unique frame definitions created: " .. unique_frame_id_counter .. ". Total frames in video: " .. NUM_FRAMES_IN_VIDEO .. ".")

-- SECTION 2: Traps
-- Purpose: Place one trap for every possible pixel position and every color that can have a trap.
-- Each trap listens on a unique `actual_pixel_activator_channel`.
print("Placing spike traps...")
bx = level.left
if unique_frame_id_counter > 0 or NUM_FRAMES_IN_VIDEO > 0 then
    by = by + 1 -- Move down one line if any relays were placed
end

-- total_pixel_channels_for_traps: Total number of distinct (pixel_position, trappable_color) combinations.
local total_pixel_channels_for_traps = num_colors_with_relays_traps * PIXELS_PER_FRAME

for relative_pixel_channel_idx = 0, total_pixel_channels_for_traps - 1 do
    -- This is the trap's receiving channel. It must match an `actual_pixel_activator_channel` from Section 1.
    local actual_trap_receive_channel = CH_OFFSET_PIXEL_ACTIVATORS + relative_pixel_channel_idx

    -- Determine the color and original pixel_index for this trap based on its relative_pixel_channel_idx
    local trap_color_block_index = math.floor(relative_pixel_channel_idx / PIXELS_PER_FRAME) -- 0 for first trappable color, 1 for second, etc.
    local pixel_index_in_frame_for_trap = relative_pixel_channel_idx % PIXELS_PER_FRAME -- 0 to (PIXELS_PER_FRAME - 1)

    -- Map trap_color_block_index back to the actual color_index from pixelColorMap and trap_color_name
    local trap_color_name = nil
    local sorted_trap_color_keys_s2 = {{}}
    for k in pairs(chaosheadColorMap) do table.insert(sorted_trap_color_keys_s2, k) end
    table.sort(sorted_trap_color_keys_s2)
    
    if sorted_trap_color_keys_s2[trap_color_block_index + 1] then
        local original_color_idx = sorted_trap_color_keys_s2[trap_color_block_index + 1]
        trap_color_name = chaosheadColorMap[original_color_idx]
    end
    
    if trap_color_name then
        place_configured_spike_trap(actual_trap_receive_channel, trap_color_name)
        
        -- Grid layout for traps:
        -- Traps are laid out in blocks, one block per color.
        -- Within each color block, they are videoData.width per row.
        if ((pixel_index_in_frame_for_trap + 1) % videoData.width) == 0 then
            bx = level.left
            by = by + 1
        end
    end
end
print("Finished placing traps.")
"""

# Function to process video frames and export color-mapped data to a Lua script file
def process_video_color_mapping_to_lua(video_filename, output_lua_filename, palette, labels,
                                     width, height, fps_target): # Renamed fps to fps_target for clarity
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
    
    # Get properties from the video *after* moviepy processing (which is what cap is reading)
    actual_fps_of_processed_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_processed_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing frames from '{video_filename}' (FPS: {actual_fps_of_processed_video:.2f}, Total frames: {total_frames_in_processed_video})")
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        # Ensure frame is resized to target_width, target_height if not already
        # OpenCV's VideoCapture should provide frames at the size of the video file.
        # If moviepy conversion was correct, frames are already target_width x target_height.
        if frame.shape[1] != width or frame.shape[0] != height:
             # This case should ideally not happen if convert_and_resize_video worked as expected.
             # If it does, resizing here is a fallback but might be slow.
            print(f"Warning: Frame {frame_count} size {frame.shape[1]}x{frame.shape[0]} does not match target {width}x{height}. Resizing.")
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        closest_indices = map_colors_to_palette_vectorized(frame, palette)
        frame_labels_chars = labels[closest_indices]
        frame_string = "".join(frame_labels_chars)
        frame_data_strings.append(frame_string)
    cap.release()
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f}s.")

    if not frame_data_strings:
        print("No frames processed.")
        return

    # The number of frames in frame_data_strings is total_frames_in_processed_video
    # The fps for Lua's videoData.fps should be fps_target (what moviepy aimed for)
    
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
        fps=fps_target, # Use the target_fps for Lua's videoData.fps
        frame_data_lua=frame_data_lua_string
    )

    output_lua_path = os.path.join(os.getcwd(), output_lua_filename)
    with open(output_lua_path, 'w') as file:
        file.write(lua_script_content)
    print(f"Lua script generated: {output_lua_path}")

if __name__ == "__main__":
    original_video_filename = "video.mp4"
    # Create a dummy video.mp4 if it doesn't exist for testing
    if not os.path.exists(original_video_filename):
        print(f"Creating a dummy '{original_video_filename}' for testing purposes.")
        dummy_width, dummy_height = 64, 48
        dummy_source_fps = 10 # Original FPS of the dummy video
        dummy_duration_seconds = 2.5 # e.g., 2.5 seconds
        dummy_total_frames = int(dummy_source_fps * dummy_duration_seconds) # 25 frames

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_dummy = cv2.VideoWriter(original_video_filename, fourcc, dummy_source_fps, (dummy_width, dummy_height))
        
        if not out_dummy.isOpened():
            print(f"Error: Could not open video writer for dummy '{original_video_filename}'. Check OpenCV/ffmpeg setup.")
        else:
            # Define a few distinct frame patterns
            frame_pattern_A = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
            frame_pattern_A[:, :dummy_width//2, 2] = 255  # Red left half

            frame_pattern_B = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
            frame_pattern_B[dummy_height//2:, :, 1] = 255  # Green bottom half
            
            frame_pattern_C = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
            cv2.circle(frame_pattern_C, (dummy_width//2, dummy_height//2), dummy_height//4, (0,0,255), -1) # Red circle

            # Create a sequence with some repetition
            sequence_patterns = [frame_pattern_A, frame_pattern_B, frame_pattern_A, frame_pattern_C, 
                                 frame_pattern_B, frame_pattern_A, frame_pattern_C, frame_pattern_C]
            
            for i in range(dummy_total_frames):
                # Cycle through the patterns to ensure some unique and some duplicate frames
                current_pattern = sequence_patterns[i % len(sequence_patterns)]
                out_dummy.write(current_pattern)
            out_dummy.release()
            print(f"Dummy '{original_video_filename}' with {dummy_total_frames} frames at {dummy_source_fps} FPS created.")

    low_res_video_filename = "video_low_res.mp4"
    color_mapped_output_lua_filename = "chaoshead_video_data.lua"
    
    target_width = 11
    target_height = 8
    target_fps = 1 # This will affect how many frames moviepy outputs for the Lua script

    if convert_and_resize_video(original_video_filename, low_res_video_filename, target_width, target_height, target_fps):
        process_video_color_mapping_to_lua(low_res_video_filename, color_mapped_output_lua_filename, palette_bgr, palette_labels, target_width, target_height, target_fps)
