import sounddevice
import pydub
from pydub.playback import play
import time
import numpy as np


class Audio:
    def __init__(self, file_path):
        self.segment = pydub.AudioSegment.from_file(file_path)
        if self.segment.channels != 1:
            self.samples = np.array(self.segment.get_array_of_samples().tolist(), dtype="int16")\
                .reshape(-1, self.segment.channels)
        else:
            self.samples = np.array(self.segment.get_array_of_samples().tolist(), dtype='int16')

    def play_sound(self):
        # sounddevice.play(self.samples, self.segment.frame_rate)
        pydub.playback.play(self.segment)

    def stop(self):
        sounddevice.stop()

    def download(self):
        self.segment.export('data/out/test.mp3', format='mp3')


if __name__ == '__main__':
    a = Audio('mokou_poke.mp4')
    # a = Audio('pk_test.mkv')
    a.play_sound()
