# Bad Apple!! in Levelhead

Scripts for converting an MP4 file into a Levelhead level. This example uses the [music video](https://archive.org/details/TouhouBadApple) for the Touhou song "Bad Apple!!". See [this level](https://www.bscotch.net/games/levelhead/levels/69fxn67) for an example of it in use.

## Video conversion

Downscales the video and reduces the framerate. Small video sizes and very low framerates (between 1 and 5) are recommended for longer videos, especially if you want to make a published level.

```
python video_conversion.py
```

## Video to array

Converts the video to a 2-dimensional array of 1s and 0s based on the brightness of each pixel. Each frame is stored in a separate array element

```
python video_to_array.py
```

## Video to Relays

[Chaoshead](https://github.com/tyoeer/chaoshead) script that can use the data from the converted array to make an animation sequence. Places a Relay for each pixel that has to be activated for each frame. To make it more efficient, duplicate frames are tracked and just reference the original frame.

Note that the videoData variable should also be updated with data from the previous script.

### Playing the video

WIP
