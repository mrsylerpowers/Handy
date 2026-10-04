"""Regenerate the decoder test fixtures in this directory.

Requires PyAV (`pip install av numpy`). Run: python make_fixtures.py .
Each file holds ~1 s of a 440 Hz sine at amplitude 0.5.
"""
import sys
import av
import numpy as np

OUT = sys.argv[1]


def tone(rate, channels):
    t = np.arange(rate) / rate
    wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.tile(wave, (channels, 1))  # planar: (channels, samples)


def write(name, fmt, codec, rate, layout, bit_rate=None, options=None, video=False):
    channels = 2 if layout == "stereo" else 1
    container = av.open(f"{OUT}/{name}", mode="w", format=fmt)
    if video:
        vstream = container.add_stream("libx264", rate=5, options={"preset": "ultrafast", "crf": "51"})
        vstream.width = 32
        vstream.height = 32
        vstream.pix_fmt = "yuv420p"
    stream = container.add_stream(codec, rate=rate, options=options or {})
    stream.layout = layout
    if bit_rate:
        stream.bit_rate = bit_rate

    if video:
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(5):
            vf = av.VideoFrame.from_ndarray(black, format="rgb24")
            vf.pts = i
            for packet in vstream.encode(vf):
                container.mux(packet)

    samples = tone(rate, channels)
    step = 1024
    for start in range(0, samples.shape[1], step):
        block = np.ascontiguousarray(samples[:, start:start + step])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
        frame.sample_rate = rate
        frame.pts = start
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    if video:
        for packet in vstream.encode(None):
            container.mux(packet)
    container.close()
    print("wrote", name)


write("tone.mp3", "mp3", "libmp3lame", 44100, "mono", bit_rate=48000)
write("tone.m4a", "ipod", "aac", 48000, "mono", bit_rate=48000)
write("tone.ogg", "ogg", "vorbis", 44100, "stereo", options={"strict": "experimental"})
write("tone.flac", "flac", "flac", 22050, "mono")
write("video.mp4", "mp4", "aac", 48000, "mono", bit_rate=48000, video=True)
write("tone.opus", "ogg", "libopus", 48000, "mono", bit_rate=24000)
write("tone.caf", "caf", "pcm_s16le", 48000, "mono")
write("tone_alac.m4a", "ipod", "alac", 44100, "mono")
