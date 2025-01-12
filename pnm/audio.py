import numpy as np
import struct
import subprocess
from scipy.signal import windows
from config import SAMPLE_RATE, N_FFT, N_SAMPLES

def read_mel_from_binary(bytes):
    reader = bytes
    rows = struct.unpack("<I", reader[0:4])[0]
    cols = struct.unpack("<I", reader[4:8])[0]
    buffer = []

    for i in range(8, len(reader), 4):
        buffer.append(struct.unpack("<f", reader[i:i+4])[0])

    return np.array(buffer).reshape((rows, cols))

def process_audio(audio, mel_filters, n_frames, from_bytes=False):
    if from_bytes:
        audio = load_audio(audio, SAMPLE_RATE)
    mel = log_mel_spectrogram(audio, mel_filters, N_SAMPLES).T
    padded_mel = pad_or_trim(mel, n_frames, axis=1)
    return np.expand_dims(padded_mel, axis=0)

def load_audio(audio_bytes, sr):
    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-nostdin",
                "-hide_banner",
                "-nostats",
                "-threads",
                "0",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=audio_bytes)
        if process.returncode != 0:
            raise Exception(f"Failed to load audio: {stderr.decode()}")

        samples_f32 = []
        for i in range(0, len(stdout), 2):
            sample = struct.unpack("<h", stdout[i:i+2])[0] / 32768.0
            samples_f32.append(sample)
        return samples_f32

    except Exception as e:
        raise Exception(f"Error loading audio: {e}")

def log_mel_spectrogram(audio, mel_filters, padding):
    extended_audio = audio[:]
    if padding > 0:
        extended_audio.extend([0.0] * padding)
    
    window = windows.hann(N_FFT)
    stft = compute_stft(extended_audio, window)

    magnitudes = np.abs(stft)**2
    magnitudes = magnitudes[:, :N_FFT // 2 + 1]
    
    mel_spec = np.dot(mel_filters, magnitudes.T).T
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    
    max_val = np.max(log_spec)
    log_spec = ((np.maximum(log_spec, max_val - 8.0)) + 4.0) / 4.0

    return log_spec


def compute_stft(audio, window):
    hop_length = N_FFT // 2
    num_frames = (len(audio) - N_FFT) // hop_length + 1
    stft = np.zeros((num_frames, N_FFT), dtype=np.complex128)

    for i in range(num_frames):
        start = i * hop_length
        end = start + N_FFT
        frame = audio[start:end] * window
        stft[i] = np.fft.fft(frame)

    return stft

def pad_or_trim(array, length, axis):
    current_length = array.shape[axis]

    if current_length < length:
       pad_width = [(0,0)] * len(array.shape)
       pad_width[axis] = (0, length - current_length)
       return np.pad(array, pad_width, mode='constant')
    elif current_length > length:
        slices = [slice(None)] * len(array.shape)
        slices[axis] = slice(0, length)
        return array[tuple(slices)]
    else:
        return array