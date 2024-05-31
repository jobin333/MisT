#!/usr/bin/python3
import sys
import torchvision

def print_keyframes(reader, duration):
  for i in range(duration):
    reader.seek(i)
    _, t = next(reader).values()
    print(t)


video_path = sys.argv[1]
print(f'Reading file {video_path}')
reader = torchvision.io.VideoReader(video_path, "video")
video_duration = reader.get_metadata()['video']['duration'][0]
video_fps = reader.get_metadata()['video']['fps'][0]
video_fps = int(video_fps)
video_duration = int(video_duration)

print(f'Video FPS: {video_fps}  Video Duration:{video_duration}')
print_keyframes(reader, 100)