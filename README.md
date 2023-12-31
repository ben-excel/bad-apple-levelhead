# Convert video files to Levelhead relays

Python and [Chaoshead](https://github.com/tyoeer/chaoshead) scripts for importing an MP4 file into a Levelhead level.

This example uses the [music video](https://archive.org/details/TouhouBadApple) for the Touhou song "Bad Apple!!". See [this level](https://www.bscotch.net/games/levelhead/levels/69fxn67) ([video](https://youtu.be/irKNagsKHsg)) for the end result.

## Video conversion

Downscales the video and reduces the framerate. Small video sizes and very low framerates (between 1 and 5) are recommended for longer videos, especially if you want to make a published level.

```
python video_conversion.py
```

## Video to array

Converts the video to a 2-dimensional array with strings of 1s and 0s based on the brightness of each pixel. Each frame is stored in a separate array element.

```
python video_to_array.py
```

## Array to Relays

Lua script for [Chaoshead](https://github.com/tyoeer/chaoshead) that can use the data from the previous converted array to make an animation sequence in a level. Places a Relay for each pixel that has to be activated for each frame.

To make it more efficient, when duplicate frames are detected it just places a Relay that references the original frame. However, be warned this script can still place a ton of Relays depending on the video's quality and frame rate and the result may be very laggy.

Note that the videoData variable should also be updated with data from the previous script.

# Playing the video

Locking Tempswitches can be used to play the video. Tempswitch time should be your `1/fps`, and path speed should `71*fps`.

<img src="https://web.archive.org/web/20231206221653/https://media.discordapp.net/attachments/809676310934192128/1182082032520667207/Screenshot_2023-12-06_151125.png" width="480">

This is how the video player could look like for a 10x12 video played on lead blocks.

<img src="https://web.archive.org/web/20231206221520/https://cdn.discordapp.com/attachments/809676310934192128/1182082558285058259/image1.png" width="480">


