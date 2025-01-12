from pnm import Pnm
from diff import compare_phonetic_strings

if __name__ == "__main__":
    with open("./test.mp3", "rb") as f:
        audio_bytes = f.read()

    pnm = Pnm()

    decoded_text, probs = pnm.generate(audio_bytes, from_bytes=True)
    print("Decoded Text:", decoded_text)
    print("Probabilities:", probs)

    

    res = compare_phonetic_strings(decoded_text, "həlˈo͡ʊ wˈɜːld hˈa͡ɪ", probs)
    print(res)
