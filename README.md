# DeepDream Animator

This tool helps to create animations with [deepdream](https://github.com/google/deepdream).
Extract frames from videos, process them with deepdream and then output as new video file.
Frame blending option is provided, to ensure "stable" dreams across frames.
A preview function make rapid iterations possible.

![deepdreamanim](https://i.imgur.com/JiMIQ01.gif "deep dream animation")

## How to
1. Extract Video  
`python dreamer.py --input myvideo/video.mp4 --output myvideo --extract 1`

2. Run DeepDream  
`python dreamer.py --input myvideo --output frames`

3. Create Video  
`python dreamer.py --input frames --output myvideo/deepdreamvideo.mp4 --create 1`

(change "myvideo" to your directory/file name)

![deepdreamanim](https://i.imgur.com/MpoYxZX.gif "deep dream animation")

## Settings
Create a preview  
`python dreamer.py --input myvideo --output myvideo/frames --preview 600 `

Tweak settings  
`python dreamer.py --input myvideo --output myvideo/frames --preview 600 --octaves 4 --octavescale 1.4 --iterations 10 --jitter 32 --zoom 1 --stepsize 1.5 --blend 0.5 --layers inception_3a/output inception_3b/output`

(Preview changes Images to width of choice, original files not changed)

(Try using multiple layers, it will cycle through them from frame to frame.)

## Batch Processing
Use the above commands and stack them by putting a ";" inbetween commands.  
`python dreamer.py --input myvideo --output myvideo/frames;python dreamer.py --input myvideo2 --output myvideo2/frames`

![deepdreamanim](https://i.imgur.com/6bpKNVL.gif "deep dream animation")

## Creative Request
It would be very helpful for other deepdream researchers, if you could include the used parameters in the description of your youtube videos. You can find the parameters in the image filenames.

## Requirements
- Python
- Caffe (and other deepdream dependencies)
- FFMPEG

## Audio
The tool currently does not handle audio.
Check out [DeepDreamVideo](https://github.com/graphific/DeepDreamVideo) by [@graphific](https://twitter.com/graphific)

![deepdreamanim](https://i.imgur.com/eH1oE6a.gif "deep dream animation")


## Credits

Samim | [Samim.io](http://samim.io) | [@samim](https://twitter.com/samim)
