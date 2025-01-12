import numpy as np
import onnxruntime as ort

from config import decode, MAX_LENGTH, stoi
from audio import process_audio, read_mel_from_binary
from utils import softmax


class Pnm:
    def __init__(
        self,
        encoder_path="./artifacts/encoder_int8.onnx",
        decoder_path="./artifacts/decoder_int8.onnx",
        mel_filters_path="./artifacts/mel_filters_80.bin",
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    ):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.mel_filters_path = mel_filters_path

        self.encoder_session = ort.InferenceSession(
            self.encoder_path, sess_options=sess_options, providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            self.decoder_path, sess_options=sess_options, providers=providers
        )

        mel_filters_80_bytes = open("./artifacts/mel_filters_80.bin", "rb").read()
        self.mel_filters_80 = read_mel_from_binary(mel_filters_80_bytes)

    def prepare_input(self, audio_bytes):
        return process_audio(audio_bytes, self.mel_filters_80, 3000)

    def inference(self, mel_segment):
        encoder_outputs = self.encoder_session.run(
            None, {"mel": mel_segment.astype(np.float32)}
        )
        context = encoder_outputs[0]

        tokens = [stoi["<"]]
        probs = [1.0]

        tokens_array = np.zeros((1, MAX_LENGTH), dtype=np.int64)
        tokens_array[0, 0] = stoi["<"]

        for i in range(1, MAX_LENGTH):
            decoder_outputs = self.decoder_session.run(
                None,
                {
                    "tokens": tokens_array[:, :i],
                    "context": context,
                },
            )
            logits = decoder_outputs[0]
            probabilities = logits[0, -1, :]
            softmax_probs = softmax(probabilities)

            idx_next = np.argmax(softmax_probs)
            prob_next = softmax_probs[idx_next]

            tokens.append(idx_next)
            probs.append(prob_next)
            tokens_array[0, i] = idx_next

            if idx_next == stoi[">"]:
                break

        if len(tokens) > 2:
            tokens = tokens[1:-1]
        if len(probs) > 2:
            probs = probs[1:-1]
        return tokens, probs

    def generate(self, audio, from_bytes=False):
        mel_segment = self.prepare_input(audio, from_bytes=from_bytes)
        tokens, probs = self.inference(mel_segment)

        decoded_output = decode(tokens)
        return decoded_output, probs
