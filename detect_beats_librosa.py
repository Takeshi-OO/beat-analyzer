# detect_beats_librosa.py
# python detect_beats_librosa.py audio\bgm1.wav  で実行可能

import sys
import librosa
import numpy as np

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

def detect_strong_onsets(audio_path, percentile_threshold=75):
    """
    強いオンセットのみを相対的な基準で検出する
    
    Parameters:
    -----------
    audio_path : str
        オーディオファイルのパス
    percentile_threshold : float
        選択するオンセットの強度の閾値（パーセンタイル）
        
    Returns:
    --------
    strong_onset_times : ndarray
        強いオンセットの時間（秒）
    onset_strengths : ndarray
        すべてのオンセットの強度
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # オンセット強度を計算
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # オンセットフレームを検出
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, 
        sr=sr
    )
    
    # 各オンセットの強度を取得
    onset_strengths = onset_env[onset_frames]
    
    # 強度の閾値を計算（パーセンタイルベース）
    threshold = np.percentile(onset_strengths, percentile_threshold)
    
    # 閾値を超えるオンセットのみを選択
    strong_onset_indices = np.where(onset_strengths >= threshold)[0]
    strong_onset_frames = onset_frames[strong_onset_indices]
    
    # フレームを時間に変換
    strong_onset_times = librosa.frames_to_time(strong_onset_frames, sr=sr)
    
    return strong_onset_times, onset_strengths

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
        
    # 強いオンセット検出（上位25%の強度を持つオンセットを選択）
    strong_onset_times, onset_strengths = detect_strong_onsets(audio_file, percentile_threshold=75)
    print("\nDetected Strong Onset Times (seconds):")
    print(f"Using threshold: top {25}% of onset strengths")
    for i, ot in enumerate(strong_onset_times):
        print(f"{ot:.3f}")
