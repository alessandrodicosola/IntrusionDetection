# = Config file
# ================================================================================================
# if True shows the sequence of transformations else shows the video
debug = False
display_video = not debug
# Start from the 0-index frame
start_from = 0
# ================================================================================================

detector_key = "background"  # background, background_no_gaussian
kind = "static"  # arg for detector['background']: static, first, adaptive

# ================================================================================================
# Insert here the absolute or relative path of the video
filename = "..\\in\\rilevamento-intrusioni-video.avi"
