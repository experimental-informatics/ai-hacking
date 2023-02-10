from moviepy.editor import *
import os
from natsort import natsorted

L = []

for root, dirs, files in os.walk("/home/student/Dokumente/ai-hacking/hacking/lisa/lisa_generation/video/video_footage/230209/poem44"):

    #files.sort()
    files = natsorted(files)
    for file in files:
        if os.path.splitext(file)[1] == '.mp4':
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("230209_output_poem44_12fps.mp4", fps=12, remove_temp=False)