import pyaudio
import os
import wave
import struct
import time
import threading
from functools import wraps


def suppress_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        save = os.dup(1), os.dup(2)
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)

        try:
            return func(*args, **kwargs)
        finally:
            os.dup2(save[0], 1)
            os.dup2(save[1], 2)
            os.close(null_fds[0])
            os.close(null_fds[1])

    return wrapper


class AudioRecorder:
    def __init__(self, sr, buffer_size=1024, device_index=None):
        self.sr = sr
        self.buffer_size = buffer_size
        self.audio_frames = []
        self.is_recording = False
        self.stop_flag = False
        self.device_index = device_index
        self.p = suppress_output(pyaudio.PyAudio)()

    def record_audio(self, duration=None):
        self.audio_frames.clear()

        if self.device_index is None:
            self.device_index = self._get_default_input_device_index()

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sr,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.buffer_size,
        )
        self.is_recording = True
        self.stop_flag = False

        def record():
            start_time = time.time()
            while self.is_recording:
                if self.stop_flag:
                    break
                data = self.stream.read(self.buffer_size)
                self.audio_frames.append(data)
                if duration and (time.time() - start_time >= duration):
                    break
            self.stream.stop_stream()
            self.stream.close()

        self.recording_thread = threading.Thread(target=record)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.stop_flag = True
        self.recording_thread.join()

    def process_audio(self):
        audio_bytes = b"".join(self.audio_frames)
        samples_f32 = []
        for i in range(0, len(audio_bytes), 2):
            sample = struct.unpack("<h", audio_bytes[i : i + 2])[0] / 32768.0
            samples_f32.append(sample)
        return samples_f32

    def save_as_wav(self, filename):
        audio_bytes = b"".join(self.audio_frames)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(audio_bytes)

    def _get_default_input_device_index(self):
        device_info = self.p.get_default_input_device_info()
        return device_info["index"]

    def close(self):
        self.p.terminate()
