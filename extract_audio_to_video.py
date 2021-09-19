#TODO test it tomorrow 

import moviepy.editor as mp
import os
import uuid

import ffmpeg

def extract_audio_to_video(video_src_path, video_dst_path, output_dir):
    clip_src = mp.VideoFileClip(video_src_path)
    print("clip_src" + str(clip_src))
    
    clip_dst = mp.VideoFileClip(video_dst_path)
    print("clip_dst" + str(clip_dst))

    audio_src_path = os.path.join(output_dir, 'audio_src.mp3')
    clip_src.audio.write_audiofile(audio_src_path)
    print("audio_src_path" + str(audio_src_path))

    audio_src = mp.AudioFileClip(audio_src_path)
    print("audio_src" + str(audio_src))

    clip_dst = clip_dst.set_audio(audio_src)
    head, tail = os.path.split(video_src_path)

    print("head, tail" + head + "," + tail)

    video_res_path = os.path.join(output_dir, "out_" + tail)
    clip_dst.write_videofile(video_res_path, codec="libx264", remove_temp=False, audio_codec='aac')

    print(str(video_res_path))

    return video_res_path

def ext_a_to_v(video_src_path, video_dst_path, output_dir):
    stream_in = ffmpeg.input(video_src_path)
    audio_in = stream_in.audio

    stream_out = ffmpeg.input(video_dst_path)

    head, tail = os.path.split(video_src_path)

    video_res_path = os.path.join(output_dir, uuid.uuid4().hex + "_" + tail)

    out = ffmpeg.output(stream_out, audio_in, video_res_path)
    out.run()

    return video_res_path