import pyaudio
import wave
import struct
import time
import threading

class AudioRecorder:
    def __init__(self, sr, buffer_size=1024, device_index=None):
        self.sr = sr
        self.buffer_size = buffer_size
        self.audio_frames = []
        self.is_recording = False
        self.stop_flag = False
        self.device_index = device_index
        self.p = pyaudio.PyAudio()

    def record_audio(self, duration=None):
        self.audio_frames.clear()

        if self.device_index is None:
            self.device_index = self._get_default_input_device_index()

        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sr,
                                  input=True,
                                  input_device_index=self.device_index,
                                  frames_per_buffer=self.buffer_size)
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
        audio_bytes = b''.join(self.audio_frames)
        samples_f32 = []
        for i in range(0, len(audio_bytes), 2):
            sample = struct.unpack("<h", audio_bytes[i:i+2])[0] / 32768.0
            samples_f32.append(sample)
        return samples_f32
    
    def save_as_wav(self, filename):
        audio_bytes = b''.join(self.audio_frames)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1) 
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(audio_bytes)

    def _get_default_input_device_index(self):
        device_info = self.p.get_default_input_device_info()
        return device_info['index']

    def close(self):
        self.p.terminate()


audio_recorder = AudioRecorder(sr=16000)
audio_recorder.record_audio(duration=10)
time.sleep(5)
audio_recorder.stop_recording()
audio_recorder.save_as_wav('recording.wav')
audio_samples = audio_recorder.process_audio()
audio_recorder.close()

print(audio_samples)

print(f"Processed {len(audio_samples)} audio samples.")
