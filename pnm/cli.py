import argparse
import pandas as pd
import os
import select
import sys
from pnm.pnm import Pnm
import time
from pnm.diff import compare_phonetic_strings
from pnm.utils import colored_print
from pnm.recorder import AudioRecorder

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import fcntl
    import tty


def get_audio_from_file(file_path):
    with open(file_path, "rb") as f:
        return f.read(), True


def _get_char():
    if sys.platform == "win32":
        return msvcrt.getch().decode("utf-8") if msvcrt.kbhit() else None
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin)
            fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
            return (
                sys.stdin.read(1)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]
                else None
            )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def get_audio_from_recorder():
    audio_recorder = AudioRecorder(sr=16000)
    audio_recorder.record_audio(duration=10)
    now = time.time()
    print("Recording... Press '<SPACE>' to stop recording.")
    while now + 10 > time.time():
        key = _get_char()
        if key == " ":
            print("Stopping recording...")
            audio_recorder.stop_recording()
            break
        elif key == "q":
            print("Quitting...")
            audio_recorder.stop_recording()
            sys.exit()
    audio = audio_recorder.process_audio()
    audio_recorder.close()
    return audio, False


def print_comparison(decoded_text, target_text, probs):
    response = compare_phonetic_strings(decoded_text, target_text, probs)

    for comparison in response["detailed_comparison"]:
        text = (
            comparison["predicted"]
            if comparison["status"] == "insertion"
            else comparison["char"]
        )
        colored_print(text or "", comparison["color"])

    average_confidence = response["metrics"]["accuracy"]
    print()
    if average_confidence < 0.3:
        colored_print(f"Confidence too low, {average_confidence:.2f}", "#ff0000")
    elif average_confidence < 0.6:
        colored_print(f"Confidence okay, {average_confidence:.2f}", "#ffff00")
    else:
        colored_print(f"Confidence good, {average_confidence:.2f}", "#00ff00")
    return response


def main():
    parser = argparse.ArgumentParser(description="PNM Audio Processing")
    subparsers = parser.add_subparsers(
        dest="option", help="Choose the processing option"
    )

    file_parser = subparsers.add_parser("file", help="Process an audio file")
    file_parser.add_argument(
        "file_path",
        type=str,
        help="Path to the audio file",
    )
    file_parser.add_argument(
        "--target_text",
        type=str,
        help="Target text to compare with (optional)",
    )

    recorder_parser = subparsers.add_parser(
        "recorder", help="Record audio from microphone"
    )
    recorder_parser.add_argument(
        "--target_text",
        type=str,
        help="Target text to compare with (optional)",
    )

    practice_parser = subparsers.add_parser(
        "practice", help="Practice session using a JSON file"
    )
    practice_parser.add_argument(
        "--practice_parquet_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "artifacts", "sample_practice.parquet"
        ),
        help="Path to the practice JSON file",
    )

    args = parser.parse_args()

    if not args.option:
        parser.print_help()
        return

    pnm = Pnm()

    if args.option == "file":
        audio, from_bytes = get_audio_from_file(args.file_path)
        decoded_text, probs = pnm.generate(audio, from_bytes=from_bytes)
        if args.target_text:
            print_comparison(decoded_text, args.target_text, probs)
        else:
            print(decoded_text)

    elif args.option == "recorder":
        audio, from_bytes = get_audio_from_recorder()
        decoded_text, probs = pnm.generate(audio, from_bytes=from_bytes)
        if args.target_text:
            print_comparison(decoded_text, args.target_text, probs)
        else:
            print(decoded_text)

    elif args.option == "practice":
        print("Starting practice session...")
        practice_data = pd.read_parquet(args.practice_parquet_file)
        practice_data = practice_data.sample(frac=1)
        stop = False
        for index, (sentence, ipa) in practice_data.iterrows():
            answering = True
            while answering:
                print()
                print(f"Read sentence: {sentence}")
                audio, from_bytes = get_audio_from_recorder()
                decoded_text, probs = pnm.generate(audio, from_bytes=from_bytes)
                print_comparison(decoded_text, ipa, probs)
                print()
                print("Try again? y/n or <SPACE> to continue | reminder: q to quit")
                while True:
                    key = _get_char()
                    if key == "y":
                        print()
                        break
                    elif key == "n" or key == " ":
                        print()
                        answering = False
                        break
                    elif key == "q":
                        stop = True
                        answering = False
                        break

            if stop:
                break


if __name__ == "__main__":
    main()
