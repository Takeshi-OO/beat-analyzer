import sys
import numpy as np
from madmom.audio import Signal
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor

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
    downbeat_times : list of tuple
        ダウンビートのタイミングと強度のタプル (秒, 強度)
    beat_times : list of tuple
        ビートのタイミングと強度のタプル (秒, 強度, ダウンビート含む)
    strong_attacks : list of tuple
        強いアタック（ビートとオンセット含む）のタイミングと強度と種類のタプル (秒, 強度, 種類)
    onset_times : list of tuple
        オンセットのタイミングと強度のタプル (秒, 強度)
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
    
    # --- 2) オンセット強度の検出 ---
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
    
    # ビートとダウンビートの情報を保持（時間と強度）
    beat_times_with_strength = []
    downbeat_times_with_strength = []
    
    for time, beat_idx in downbeat_info:
        # オンセット強度を取得
        frame_idx = int(time * fps)
        strength = 0.0
        if 0 <= frame_idx < len(onset_activation):
            strength = onset_activation[frame_idx]
        
        if beat_idx == 1:  # ダウンビート
            downbeat_times_with_strength.append((time, strength))
        beat_times_with_strength.append((time, strength))
    
    # 全てのオンセットの強度を取得
    all_onsets_with_strength = []
    for onset_time in onset_times:
        frame_idx = int(onset_time * fps)
        if 0 <= frame_idx < len(onset_activation):
            onset_strength = onset_activation[frame_idx]
            all_onsets_with_strength.append((onset_time, onset_strength))
    
    # ビートとオンセットを統合（重複を考慮）
    all_attacks = []
    beat_times_array = np.array([time for time, _ in beat_times_with_strength])
    
    # まずビートを追加
    for time, strength in beat_times_with_strength:
        is_downbeat = any(abs(db_time - time) < tolerance for db_time, _ in downbeat_times_with_strength)
        attack_type = "downbeat" if is_downbeat else "beat"
        all_attacks.append((time, strength, attack_type))
    
    # 次にビートと重複しないオンセットを追加
    for time, strength in all_onsets_with_strength:
        # 既存のビートと重複していないか確認
        if np.all(np.abs(beat_times_array - time) > tolerance):
            all_attacks.append((time, strength, "onset"))
    
    # 強度でソート（降順）
    all_attacks.sort(key=lambda x: x[1], reverse=True)
    
    # ダウンビートの数と同じ数だけ強いアタックを選出
    strong_attacks = []
    if len(downbeat_times_with_strength) > 0:
        num_strong_attacks = len(downbeat_times_with_strength)
        strong_attacks = all_attacks[:num_strong_attacks]
    
    # 時間順にソート
    strong_attacks.sort(key=lambda x: x[0])
    
    # オンセットのみのリストも作成（ビートと重複しないもの）
    onset_times_with_strength = []
    for time, strength in all_onsets_with_strength:
        if np.all(np.abs(beat_times_array - time) > tolerance):
            onset_times_with_strength.append((time, strength))
    
    return downbeat_times_with_strength, beat_times_with_strength, strong_attacks, onset_times_with_strength

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "audio/bgm1.wav"  # 解析したいオーディオファイルへのパス
    
    # 解析パラメータ
    tolerance = 0.02  # ビートとオンセットが重複とみなす時間差（秒）
    
    # ビート、ダウンビート、強いアタック、オンセットを検出
    downbeats, beats, strong_attacks, onsets = detect_beats_and_strong_attacks(audio_file)
    
    print(f"ダウンビート数: {len(downbeats)}")
    print(f"ビート数（ダウンビート含む）: {len(beats)}")
    print(f"オンセット数（ビートと重複しないもの）: {len(onsets)}")
    print(f"強い音響イベント数（ビート・オンセット含む、強度順）: {len(strong_attacks)}")
    
    # 強い音響イベントの内訳を計算
    strong_downbeats = sum(1 for _, _, type_ in strong_attacks if type_ == "downbeat")
    strong_beats = sum(1 for _, _, type_ in strong_attacks if type_ == "beat")
    strong_onsets = sum(1 for _, _, type_ in strong_attacks if type_ == "onset")
    
    print(f"  - うちダウンビート: {strong_downbeats}")
    print(f"  - うち通常ビート: {strong_beats}")
    print(f"  - うちオンセット: {strong_onsets}")
    
    # 時系列順に全てのイベントを表示するための準備
    all_events = []
    
    # ダウンビートを追加（小節番号付き）
    for i, (time, strength) in enumerate(downbeats):
        all_events.append((time, f"ダウンビート", i+1, strength))
    
    # 通常のビート（ダウンビート以外）を追加
    downbeat_times = [time for time, _ in downbeats]
    for time, strength in beats:
        if time not in downbeat_times:  # ダウンビートでないビートのみ追加
            # 該当する小節番号を計算
            bar_number = 0
            for i, (db_time, _) in enumerate(downbeats):
                if time > db_time:
                    bar_number = i + 1
                else:
                    break
            
            all_events.append((time, "ビート", bar_number, strength))
    
    # オンセットを追加（強い音響イベントに選出されたもののみ）
    strong_onset_times = [time for time, _, type_ in strong_attacks if type_ == "onset"]
    
    for time, strength in onsets:
        if any(abs(time - st) < tolerance for st in strong_onset_times):
            # 該当する小節番号を計算
            bar_number = 0
            for i, (db_time, _) in enumerate(downbeats):
                if time > db_time:
                    bar_number = i + 1
                else:
                    break
            
            all_events.append((time, "オンセット", bar_number, strength))
    
    # 強い音響イベントをマークする
    strong_event_times = [time for time, _, _ in strong_attacks]
    
    # 時間順にソート
    all_events.sort(key=lambda x: x[0])
    
    print("\n=== 時系列順のイベント ===")
    for time, event_type, bar_number, strength in all_events:
        # 強い音響イベントかどうかをマーク
        is_strong = any(abs(time - st) < tolerance for st in strong_event_times)
        strength_label = "1" if is_strong else "0"
        bar_str = f"{bar_number:03d}" if bar_number > 0 else "000"
        
        # 時間を四捨五入して小数点1桁にし、0埋め
        time_str = f"{time:.1f}".zfill(5)
        
        print(f"時間: {time_str} 秒, 小節: {bar_str}, 強度: {strength:.2f}, 強弱: {strength_label}, タイプ: {event_type}")
