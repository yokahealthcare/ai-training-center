from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# CCTV FIGHT - FIGHT
# need = [
#     [190, 230],
#     [250, 305],
#     [355, 390],
#     [460, 505],
#     [595, 640],
# ]
# CCTV FIGHT - NO FIGHT
need = [
    [775, 825],
    [910, 965],
    [1010, 1095],
    [1190, 1275]
]



# CCTV FIGHT MASJID - FIGHT
# need = [
#     [195, 230],
#     [240, 290],
#     [405, 440],
#     [455, 490],
#     [505, 545]
# ]

for start_time, end_time in need:
    filename = f"NO_FIGHT_{start_time}_{end_time}.mp4"
    filepath = f"lapas ngaseman/CCTV FIGHT/{filename}"
    ffmpeg_extract_subclip("lapas ngaseman/CCTV FIGHT.mp4", start_time, end_time, targetname=filepath)
