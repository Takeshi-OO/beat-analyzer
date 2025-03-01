import sys
import numpy as np
from madmom.audio import Signal
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor

def detect_downbeats(audio_file):
    """
    指定したオーディオファイルからダウンビートを検出し、結果を返す。
    
    Returns:
        downbeats: [(time_sec, beat_type), ...] のリスト
                   beat_type=1: ダウンビート, 2: 通常のビート
    """
    # RNNを用いてアクティベーションを推定
    # fps=100 はフレームレート。処理速度と精度のバランスによって変更可能
    rnn_processor = RNNDownBeatProcessor(fps=100)
    activation = rnn_processor(audio_file)
    
    # DBNを用いて最終的にビート / ダウンビートを推定
    # beats_per_bar=[4] は1小節あたりのビート数(4拍子)を仮定した設定
    # 他の拍子の場合は [3] や [4, 3] (複数候補) などを試す
    dbn_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
    # 出力形式: [(time_sec, beat_type), ...]
    # beat_type=1: 小節頭(ダウンビート), 2: それ以外のビート
    downbeats = dbn_processor(activation)
    
    return downbeats

def detect_beats_and_strong_attacks(audio_file, fps=100, beat_per_bar=4, rel_threshold_factor=0.3, tolerance=0.02):
    """
    指定した音声ファイルに対して、
    1. ビート・ダウンビートの検出
    2. 相対閾値を用いた強いアタック（オンセット）の検出
    を行い、それぞれのタイミングを返す。
    
    Parameters
    ----------
    audio_file : str
        入力音声ファイルのパス (wavなど)
    fps : int, optional
        分析に用いるフレームレート(デフォルト=100)
    beat_per_bar : int or list, optional
        小節あたりの拍数。4拍子なら4(デフォルト=4)
    rel_threshold_factor : float, optional
        オンセット検出の相対閾値係数（最大値に対する割合、デフォルト=0.3）
    tolerance : float, optional
        ビートとオンセットが重複とみなす時間差（秒、デフォルト=0.02）

    Returns
    -------
    downbeat_times : list of float
        ダウンビートのタイミング（秒）
    beat_times : list of float
        ビートのタイミング（秒, ダウンビート含む）
    strong_attacks : list of float
        ビート外を含む強いアタックのタイミング（秒）
    """
    
    # --- 1) ビート・ダウンビートの検出 ---
    # RNNの推定器
    downbeat_proc = RNNDownBeatProcessor(fps=fps)
    # DBNトラッカー
    downbeat_tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=[beat_per_bar] if isinstance(beat_per_bar, int) else beat_per_bar,
        fps=fps
    )
    
    # 音声ファイルからダウンビート情報を取得
    db_activation = downbeat_proc(audio_file)
    downbeat_info = downbeat_tracker(db_activation)
    
    beat_times = []
    downbeat_times = []
    for time, beat_idx in downbeat_info:
        if beat_idx == 1:  # ダウンビート
            downbeat_times.append(time)
        beat_times.append(time)
    
    # --- 2) 強いアタック（オンセット）の検出 ---
    # オンセット強度をRNNで推定
    onset_proc = RNNOnsetProcessor(fps=fps)
    onset_activation = onset_proc(audio_file)
    
    # 相対閾値を決める
    max_activation = np.max(onset_activation)
    rel_threshold = rel_threshold_factor * max_activation  # 相対閾値: 最大値の指定割合
    
    # ピークピッキング
    peak_picker = OnsetPeakPickingProcessor(
        threshold=rel_threshold,
        fps=fps,
        pre_avg=0.01,    # ピーク検出時の平均化に用いる前方時間(秒)
        post_avg=0.01,   # ピーク検出時の平均化に用いる後方時間(秒)
        pre_max=0.01,    # ピーク検出時の最大値検索窓 (秒)
        post_max=0.01    # ピーク検出時の最大値検索窓 (秒)
    )
    
    # オンセット時刻を取得
    onset_times = peak_picker(onset_activation)
    
    # ビート上と重なるオンセットを除外
    strong_attacks = []
    beat_times_array = np.array(beat_times)
    
    for onset_time in onset_times:
        # beat_times に近いタイミングがあるか判定
        if np.all(np.abs(beat_times_array - onset_time) > tolerance):
            strong_attacks.append(onset_time)
    
    return downbeat_times, beat_times, strong_attacks

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "audio/bgm1.wav"  # 解析したいオーディオファイルへのパス
    
    # 従来の方法でダウンビートとビートを検出
    results = detect_downbeats(audio_file)
    
    print("=== 従来の方法（ビートとダウンビートのみ）===")
    for (beat_time, beat_type) in results:
        beat_label = "ダウンビート" if beat_type == 1 else "ビート"
        print(f"時間: {beat_time:.2f} 秒, タイプ: {beat_label}")
    
    # 新しい方法でビート、ダウンビート、強いアタックを検出
    print("\n=== 新しい方法（ビート外の強いアタックも検出）===")
    downbeats, beats, strong_attacks = detect_beats_and_strong_attacks(audio_file)
    
    print(f"ダウンビート数: {len(downbeats)}")
    print(f"ビート数（ダウンビート含む）: {len(beats)}")
    print(f"ビート外の強いアタック数: {len(strong_attacks)}")
    
    print("\nダウンビートのタイミング:")
    for time in downbeats:
        print(f"  {time:.2f} 秒")
    
    print("\nビートのタイミング:")
    for time in beats:
        print(f"  {time:.2f} 秒")
    
    print("\nビート外の強いアタックのタイミング:")
    for time in strong_attacks:
        print(f"  {time:.2f} 秒")
