-- Script to import import.mid as boomboxes with repeat optimization

-- MIDI note mapping
local midiNotes = {
	"C",false, -- C
	"C",true,  -- C#
	"D",false, -- D
	"D",true,  -- D#
	"E",false, -- E
	"F",false, -- F
	"F",true,  -- F#
	"G",false, -- G
	"G",true,  -- G#
	"A",false, -- A
	"A",true,  -- A#
	"B",false, -- B
}

-- Percussion note mapping (MIDI note number to Boombox Percussion Note name)
-- These are the common General MIDI drum map notes (channel 10, index 9)
local perc = {
	["Kick Deep"] = {35}, -- B-1
	["Kick Thump"] = {36}, -- C0
	["Snare Tough"] = {37,38,40}, -- C#0, D0, E0
	["Snare Slap"] = {66}, -- F#2
	["Snare Reverb"] = {51}, -- D#1
	["Hi-Hat Tap"] = {80,44}, -- G#4, F#0
	["Hi-Hat Close"] = {81,42}, -- A4, F0
	["Hi-Hat Open"] = {59,46}, -- B1, A#0
	["Crash"] = {52,49,57}, -- E1, C#1, A1
	["Tom Low"] = {45,47}, -- G0, B0
	["Tom High"] = {48,50}, -- C1, D1
	["Tom Low Soft"] = {41}, -- F#0
	["Tom High Soft"] = {43,60}, -- G#0, C2
	["Click"] = {39,82}, -- D#0, B4 (often metronome clicks)
	["Click Soft"] = {69}, -- A#2
}

-- Required module for MIDI parsing (assuming 'libs.midi' or 'scripts.midi')
-- Adjust the path as needed based on where you placed the midi.lua file
local M = require("scripts.midi") -- Example path

-- Read the MIDI file from user data
-- Ensure your import.mid file is in Chaoshead's 'scripts' directory
local raw, nRaw = love.filesystem.read("scripts/import.mid")

if not raw then
    print("Error: Could not read scripts/import.mid")
    print("Please place your MIDI file at 'CheeseHeist/scripts/import.mid' in your LOVE user data folder.")
    -- You might want to exit the script or return here if running directly
    -- return
end

local score = M.midi2score(raw)

if not score then
    print("Error: Could not parse MIDI data from scripts/import.mid")
    -- return -- Or handle parse error appropriately
end

-- midi2score returns { ticks_per_beat, {track1_events}, {track2_events}, ... }
local ticksPerBeat = score[1]

-- Global state needed for timing, repeat logic, and placement

local tempo = nil -- Will be set by the first set_tempo event
local latestBoombox = {
	Melody = {},
	Bass = {},
	Percussion = {},
}

-- Placement coordinates for new boomboxes (using the first script's sequential method)
-- Requires 'level' global access, which is standard in Cheese Heist scripts
local bx = level.left
local by = level.top

-- Keep track of percussion notes that don't map to a defined type
local missing_percussion = {}


-- Helper functions

-- Converts MIDI ticks (from start of score or previous event) to beats
local function ticksToBeat(ticks)
	-- Ensure ticksPerBeat is set and valid
	if not ticksPerBeat or ticksPerBeat <= 0 then
		print("Error: ticksPerBeat not set or invalid! Cannot convert ticks to beats.")
		return 0 -- Or handle error
	end
	return ticks / ticksPerBeat
end

-- Maps a MIDI note number to a Percussion Note name string
local function midiToPerc(midi)
	for k,v in pairs(perc) do
		for _,vv in ipairs(v) do
			if vv==midi then
				return k -- Return the percussion name
			end
		end
	end
	return nil -- Not found
end

-- Maps a MIDI note number to a Note Pitch string (e.g., "C4", "D#3")
local function midiToNote(midi)
	-- MIDI note 0 is C-1
	local octave = math.floor(midi/12) - 1
	local i = (midi%12) -- Index within the 12 notes of an octave (0-11)
	-- Index into midiNotes is 2*i + 1 for the note name
    -- Need to be careful with index out of bounds, though MIDI notes > 127 are invalid
    if i*2+1 > #midiNotes then
         print(string.format("Warning: MIDI note %i results in unexpected index %i for midiNotes.", midi, i*2+1))
         return "C0" -- Default or error value
    end
	local noteName = midiNotes[2*i+1]
	return noteName .. octave
end

-- Checks if a MIDI note number corresponds to a sharp/flat note
local function isSharp(midi)
	local i = (midi%12)
	-- Index into midiNotes is 2*i + 2 for the sharp boolean
    if i*2+2 > #midiNotes then
         print(string.format("Warning: MIDI note %i results in unexpected index %i for sharp check.", midi, i*2+2))
         return false -- Default or error value
    end
	return midiNotes[2*i+2] or false -- Ensure it's a boolean
end

-- Converts MIDI velocity (0-127) to Boombox volume (0-100)
local function velocityToVolume(velocity)
	-- Clamp velocity to expected range just in case
	velocity = math.max(0, math.min(127, velocity))
	return math.floor(velocity / 127 * 100) -- Use math.floor for integer volume
end

-- Function to configure a boombox object
-- This function *configures* an already placed object, it doesn't place it.
local function configureBoombox(b, note, startDelayBeats, durationBeats, volume, isPerc)
	local instrument -- Will hold "Melody", "Bass", or "Percussion"

	if isPerc then
		instrument = "Percussion"
		-- midiToPerc check was already done in processNoteEvent, safe to call here
		b:setPercussionNote(midiToPerc(note))
		b:setInstrument("Percussion")
	else
		-- Determine Melody or Bass based on pitch range
		-- Assuming 48 is C4 (often middle C or C above), common split point
		if note >= 48 then
			instrument = "Melody"
			b:setMelodyPitch(midiToNote(note))
			b:setInstrument("Melody")
		else
			instrument = "Bass"
			b:setBassPitch(midiToNote(note))
			b:setInstrument("Bass")
		end
	end

	if isSharp(note) then
		b:setSharp("Yes")
	else
		b:setSharp("No")
	end

	-- Timing and Volume
	-- Ensure tempo is set before assigning to boombox
	if tempo then
		b:setBeatsPerMinute(tempo)
	else
		-- Fallback or warning if tempo wasn't found in MIDI before the first note
		-- This might happen if the first event is a note, not set_tempo
		print("Warning: Tempo not found when configuring boombox! Using default (120).")
		b:setBeatsPerMinute(120) -- Or some default
	end

	-- StartDelayBeats property expects a string "No Delay" or a number string "X.Y"
	-- Clamp to a reasonable maximum
	local maxStartDelay = 999999 -- Example limit
	if startDelayBeats < 0 then startDelayBeats = 0 end -- Should not happen with MIDI start times
	if startDelayBeats > maxStartDelay then
		print(string.format("Warning: Start Delay %.2f exceeds max (%i). Clamping.", startDelayBeats, maxStartDelay))
		startDelayBeats = maxStartDelay
	end

	if startDelayBeats == 0 then
		b:setStartDelayBeats("No Delay")
	else
		b:setStartDelayBeats(tostring(startDelayBeats))
	end

	-- NoteBeats property expects a string "No Note" or a number string "X.Y"
	-- Clamp to a reasonable maximum, and handle zero duration
	local maxNoteBeats = 999 -- Example limit
	if durationBeats <= 0 then
		b:setNoteBeats("No Note") -- Or handle very short notes differently?
		print(string.format("Warning: Note duration is zero or negative for note %s at beat %s. Setting NoteBeats to 'No Note'.", isPerc and midiToPerc(note) or midiToNote(note), tostring(startDelayBeats)))
	else
		if durationBeats > maxNoteBeats then
			print(string.format("Warning: Note Duration %.2f exceeds max (%i). Clamping.", durationBeats, maxNoteBeats))
			durationBeats = maxNoteBeats
		end
		b:setNoteBeats(tostring(durationBeats))
	end

	-- Volume property expects a number 0-100
	b:setVolume(volume)

	-- Other common boombox properties (can be customized)
	b:setReceivingChannel(999) -- Example channel
	b:setSwitchRequirements("Any Inactive") -- Example requirement
	b:setInvisible("Yes") -- Often set to invisible for purely musical elements

	-- Update the tracking table for repeat logic
	-- This needs to happen *after* configuring the boombox
	latestBoombox[instrument][note] = b

	-- Optional: Add the placed boombox to the current selection
	-- This requires the 'selection' global and a mask or similar tool
	-- If you don't have selection tools, comment this out.
	-- if selection and selection.mask then
	-- 	selection.mask:add(b.x, b.y)
	-- end
end

-- Function to handle a single note event, deciding whether to place a new boombox
-- or extend the repeat of a previous one.
local function processNoteEvent(note, startTimeTicks, durationTicks, channel, velocity)
	-- Convert timing to beats
	local startDelayBeats = ticksToBeat(startTimeTicks)
	local durationBeats = ticksToBeat(durationTicks)
	local volume = velocityToVolume(velocity)
	local isPerc = (channel == 9) -- MIDI channel 10 (index 9) is typically percussion

	local instrument -- Declare instrument here, will be assigned below

	if isPerc then
		-- Check if percussion note is mapped
		if not midiToPerc(note) then
			-- Add to missing list and skip placing/processing this note
			table.insert(missing_percussion, note)
			-- print(string.format("Skipping unmapped percussion note: MIDI %i at beat %.2f", note, startDelayBeats)) -- Optional debug
			return -- Skip this note
		end
		instrument = "Percussion"
	else
		-- Check note pitch range
		-- Assuming Cheese Heist supports notes from MIDI 33 (A0) to 96 (C6)
		-- You might need to adjust these limits based on game capabilities
		if note < 33 then
			print(string.format("Note %s at beat %.2f too low (MIDI %i)! Skipping.", midiToNote(note), startDelayBeats, note))
			return -- Skip this note
		elseif note > 96 then
			print(string.format("Note %s at beat %.2f too high (MIDI %i)! Skipping.", midiToNote(note), startDelayBeats, note))
			return -- Skip this note
		end
		-- Determine Melody or Bass for tracking based on pitch range
		if note >= 48 then -- C4 and above are Melody
			instrument = "Melody"
		else -- Below C4 are Bass
			instrument = "Bass"
		end
	end

	-- --- Repeat Logic ---
	-- Look up the last placed boombox for this specific instrument and note/pitch
	local prev = latestBoombox[instrument][note]

	-- Check if there was a previous boombox for this instrument/note AND
	-- if the current note has the same duration and volume
	-- Use a small tolerance for comparing floating-point beat durations
	local float_tolerance = 0.001
	local prevDurationStr = prev and prev:getNoteBeats() or "No Note"
	local prevDurationBeats = (prevDurationStr == "No Note") and 0 or tonumber(prevDurationStr) or 0

	if prev and math.abs(prevDurationBeats - durationBeats) < float_tolerance and prev:getVolume() == volume then
		local prevStartDelayStr = prev:getStartDelayBeats()
		local prevStartDelay = (prevStartDelayStr == "No Delay") and 0 or tonumber(prevStartDelayStr) or 0

		local prevRepeatBeatsStr = prev:getRepeatBeats()
		local prevRepeatBeats = (prevRepeatBeatsStr == "No Repeat") and 0 or tonumber(prevRepeatBeatsStr) or 0

		local prevRepeatCount = prev:getRepeatCount() or 1 -- Default count is 1 if not repeating

		-- Calculate the time difference between the current note and the *start* of the previous note
		local delayFromPrevStart = startDelayBeats - prevStartDelay

		-- Check if the previous note was not repeating yet (RepeatBeats is 0 or "No Repeat")
		if prevRepeatBeats == 0 then
			-- Check if the delay is positive (current note starts after previous one)
			-- and not too small (avoid simultaneous notes or notes very close together causing repeats)
			if delayFromPrevStart > float_tolerance then
				-- This note is the second note in a potential repeat sequence
				-- Set the repeat interval to the time between the first two notes
				prev:setRepeatBeats(tostring(delayFromPrevStart))
				prev:setRepeatCount(2) -- Now there are 2 occurrences (original + this one)
				-- print(string.format("Extended note %s at beat %.2f to repeat with interval %.2f", isPerc and midiToPerc(note) or midiToNote(note), prevStartDelay, delayFromPrevStart)) -- Optional debug
				return -- Successfully extended, no new boombox needed
			end
		-- Check if the previous note was already repeating (RepeatBeats > 0 or "X.Y")
		elseif prevRepeatBeats > 0 then
			-- Calculate the start time of where the *next* note *should* be in the repeat sequence
			local expectedNextBeat = prevStartDelay + prevRepeatCount * prevRepeatBeats -- Count is 1 + number of repeats AFTER the first note

			-- Check if the current note starts approximately at the expected next beat
			if math.abs(startDelayBeats - expectedNextBeat) < float_tolerance then
				-- This note continues the existing repeat sequence
				prev:setRepeatCount(prevRepeatCount + 1) -- Increment the count
				-- print(string.format("Extended repeat for note %s at beat %.2f (now count %i)", isPerc and midiToPerc(note) or midiToNote(note), prevStartDelay, prevRepeatCount + 1)) -- Optional debug
				return -- Successfully extended, no new boombox needed
			end
			-- If the timing doesn't match the existing repeat interval, this note doesn't fit
		end
		-- If we reach here, the current note does *not* fit the repeat pattern
		-- either because it's simultaneous (delayFromPrevStart <= tolerance),
		-- has a different interval, or the previous note wasn't eligible (different duration/volume)

	end
	-- --- End Repeat Logic ---

	-- If we couldn't extend a repeat sequence, place a new boombox
	-- Check if there's space before placing
	if by > level.bottom then
		print("Warning: Ran out of vertical space to place boomboxes at y="..by.."! Skipping placement for note "..note.." at beat "..startDelayBeats)
		-- We still update latestBoombox *after* the call if configureBoombox happens,
		-- but since we're skipping placement, we should probably *not* update latestBoombox
		-- with a non-existent object. Let's skip configuring and updating latestBoombox too.
		return
	end

	local b = level:placeBoombox(bx, by) -- Place a new object at the next available spot

	-- Configure the new boombox object
	configureBoombox(b, note, startDelayBeats, durationBeats, volume, isPerc)

	-- Move to the next placement spot for the *next* new object
	bx = bx + 1
	if bx > level.right then
		bx = level.left
		by = by + 1
		-- Vertical space check is done at the beginning of the function now
	end
end

-- Main Processing Loop

-- score[2] is the first track, score[#score] is the last
-- Iterate through each track (starting from index 2 as score[1] is ticks_per_beat)
for trackIndex = 2, #score do
	local trackEvents = score[trackIndex]
	if not trackEvents then
		print(string.format("Warning: Skipping invalid track at index %i", trackIndex))
		goto continue_track -- Use goto to skip to the next track
	end

	-- Iterate through each event in the track
	for eventIndex, event in ipairs(trackEvents) do
		-- Basic check for event structure
		if type(event) ~= "table" or #event < 2 then
			print(string.format("Warning: Skipping invalid event in track %i at index %i", trackIndex, eventIndex))
			goto continue_event -- Use goto to skip to the next event
		end

		local eventType = event[1]
		local startTimeTicks = event[2] -- Common field for timing

		if eventType == "set_tempo" then
			-- Only capture the *first* tempo event found. MIDI files can have multiple,
			-- but Cheese Heist boomboxes only support a single BPM.
			-- event format: { "set_tempo", startTimeTicks, tempo_microseconds_per_beat }
			if tempo == nil and #event >= 3 and type(event[3]) == "number" then
				local midiTempoMicrosecondsPerBeat = event[3]
				-- BPM = 60 / seconds_per_beat
				-- seconds_per_beat = microseconds_per_beat / 1,000,000
				if midiTempoMicrosecondsPerBeat > 0 then
					tempo = 60 / (midiTempoMicrosecondsPerBeat / 1000000)
					print(string.format("Found tempo: %.2f BPM at tick %i", tempo, startTimeTicks))
					-- Optional: Set the level's global BPM here if you want
					-- level:setBeatsPerMinute(tempo)
				else
					print(string.format("Warning: Skipping set_tempo event with invalid microseconds per beat (%i) at tick %i", midiTempoMicrosecondsPerBeat, startTimeTicks))
				end
			-- else
			-- 	if tempo ~= nil then
			-- 		-- print(string.format("Ignoring subsequent set_tempo event at tick %i (tempo already set to %.2f)", startTimeTicks, tempo)) -- Optional debug
			-- 	else
			-- 		print(string.format("Warning: Skipping malformed set_tempo event in track %i at index %i", trackIndex, eventIndex))
			-- 	end
			end
		elseif eventType == "note" then
			-- event format: { "note", startTimeTicks, durationTicks, channel, noteNumber, velocity }
			if #event >= 6 and type(event[3]) == "number" and type(event[4]) == "number" and type(event[5]) == "number" and type(event[6]) == "number" then
				local durationTicks = event[3]
				local channel = event[4]
				local noteNumber = event[5] -- MIDI note number (0-127)
				local velocity = event[6]   -- MIDI velocity (0-127)

                -- Ensure tempo is set before processing notes that rely on ticksToBeat meaningfully
                -- If tempo wasn't set by a set_tempo event, use a default
                if tempo == nil then
                     print("Warning: Tempo not set by MIDI file. Using default (120 BPM) for timing calculations.")
                     tempo = 120
                end
                if ticksPerBeat == nil or ticksPerBeat <= 0 then
                    print("Error: ticksPerBeat is not set or invalid! Cannot process notes.")
                    break -- Exit loop if timing is fundamentally broken
                end


				-- Process the note event using the function with repeat logic
				processNoteEvent(noteNumber, startTimeTicks, durationTicks, channel, velocity)
			-- else
			-- 	print(string.format("Warning: Skipping malformed note event in track %i at index %i", trackIndex, eventIndex)) -- Optional debug malformed events
			end
		-- Add other event types here if needed (e.g., pitch_bend, control_change, etc.)
		-- else
		-- 	print("Skipping unsupported event type:", eventType) -- Optional debug unsupported types

		end
		::continue_event:: -- Label for goto
	end
	::continue_track:: -- Label for goto
end

-- Print any percussion notes that weren't mapped
table.sort(missing_percussion)
local missing_output = ""
local had_missing = {}
for _, note_num in ipairs(missing_percussion) do
	if not had_missing[note_num] then
		-- Try to convert to note name for better readability, but keep number if no mapping
		-- Ensure midiToNote handles the range correctly, though these are likely > 32
		local note_name = "MIDI"..note_num
        if note_num >=0 and note_num <= 127 then
             note_name = midiToNote(note_num)
        end
		missing_output = missing_output .. note_name .. "("..note_num.."), " -- Added space for clarity
		had_missing[note_num] = true
	end
end
if missing_output ~= "" then
	-- Remove trailing comma and space
	missing_output = missing_output:sub(1, -3)
	print("Warning: Percussion notes not found in 'perc' mapping (skipped placement): "..missing_output)
end

print("MIDI import process finished.")
if tempo then
    print(string.format("Final Tempo used: %.2f BPM", tempo))
end
-- print(string.format("Total boomboxes placed (including repeats as one object): %i", level:countObjects("Boombox")))

-- Optional: Select the placed objects (requires selection tools)
-- You would need to initialize 'selection' and 'selection.mask' earlier if they aren't globals
-- Example (needs proper tool setup):
-- if selection and selection.mask then
-- 	selection:update() -- Rebuild selection based on mask
-- 	print("Placed objects selected.")
-- end
