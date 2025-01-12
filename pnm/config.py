from utils import exact_div

n_mels = 80
n_ctx = 1500
n_state = 512
n_head = 8
n_layer = 6

MAX_LENGTH = 256
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)

block_size = 448
vocab = [
    "~",
    " ",
    "a",
    "b",
    "d",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "z",
    "æ",
    "ç",
    "ð",
    "ŋ",
    "ɐ",
    "ɑ",
    "ɔ",
    "ə",
    "ɚ",
    "ɛ",
    "ɜ",
    "ɡ",
    "ɪ",
    "ɬ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʊ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ʲ",
    "ˈ",
    "ˌ",
    "ː",
    "̃",
    "̩",
    "͡",
    "θ",
    "ᵻ",
    "<",
    ">",
]
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(idxs):
    return "".join([itos[idx] for idx in idxs])
