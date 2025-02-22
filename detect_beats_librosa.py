# detect_beats_librosa.py

import sys
import librosa

def detect_beats(audio_path):
    # オーディオ読み込み
    y, sr = librosa.load(audio_path, sr=None)
    # BPM(テンポ)とビートフレームを推定
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # フレームを秒単位に変換
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return tempo, beat_times

def detect_onsets(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_beats_librosa.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # ビート検出
    tempo, beat_times = detect_beats(audio_file)
    print(f"Detected Tempo: {float(tempo):.2f} BPM")
    print("Detected Beat Times (seconds):")
    for t in beat_times:
        print(f"{t:.3f}")

    # オンセット検出
    onset_times = detect_onsets(audio_file)
    print("Detected Onset Times (seconds):")
    for ot in onset_times:
        print(f"{ot:.3f}")
