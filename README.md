# pnm

pnm is an audio-to-phoneme conversion tool designed to transform spoken English into phonetic transcriptions. This project is a mini-project derived from a larger, unfinished personal project aimed at creating a tool for English phonetic practice. Although the main project wasn't completed, PNM is being transformed into a Python library for open-source use.

Currently, the tool is a work-in-progress but is functional and offers a simple way to convert audio into phonemes.

It is possible to classify the speech quality of the person training using the pnm tool. By analyzing the phonetic transcriptions generated from the spoken audio (by token confidence). This analysis can help in evaluating the quality of the speaker’s pronunciation and progress over time, allowing for personalized feedback during training. 

## Installation

To install the required dependencies, use the following command:

For cpu

```bash
pip install "pnm[cpu]"
```

For cuda 11.X
```bash
pip install "pnm[gpu]"
```

For cuda 12.X
```bash
pip install "pnm[gpu]" --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## Usage

### Command Line Interface

For get the phonemes of an audio file:

```bash
pnm file --file_path path/to/audio.wav
```

For get the phonemes of an audio recorder (default input device):

```bash
pnm recorder
```

For practice (default input device):

```bash
pnm practice
```

# Images

<p align="center">
  <img src="https://raw.githubusercontent.com/pstwh/pnm/main/examples/image.png" width="768" />
</p>
