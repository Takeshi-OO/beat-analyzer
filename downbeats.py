import sys
import numpy as np
from madmom.audio import Signal
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

def detect_beats_and_downbeats(audio_file, fps=100, beat_per_bar=4):
    """
    指定した音声ファイルに対して、ビート・ダウンビートの検出を行い、
    それぞれのタイミングを返す。
    
    Parameters
    ----------
    audio_file : str
        入力音声ファイルのパス (wavなど)
    fps : int, optional
        分析に用いるフレームレート(デフォルト=100)
    beat_per_bar : int or list, optional
        小節あたりの拍数。4拍子なら4(デフォルト=4)

    Returns
    -------
    downbeat_times : list
        ダウンビートのタイミング (秒)
    beat_times : list
        ビートのタイミング (秒, ダウンビート含む)
    """
    
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
    
    # ビートとダウンビートの情報を保持
    beat_times = []
    downbeat_times = []
    
    for time, beat_idx in downbeat_info:
        if beat_idx == 1:  # ダウンビート
            downbeat_times.append(time)
        beat_times.append(time)
    
    return downbeat_times, beat_times

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "audio/bgm1.wav"  # 解析したいオーディオファイルへのパス
    
    # ビート、ダウンビートを検出
    downbeats, beats = detect_beats_and_downbeats(audio_file)
    
    print(f"ダウンビート数: {len(downbeats)}")
    print(f"ビート数（ダウンビート含む）: {len(beats)}")
    print(f"小節数: {len(downbeats)}")
    
    # 時系列順に全てのイベントを表示するための準備
    all_events = []
    
    # ダウンビートを追加（小節番号付き）
    for i, time in enumerate(downbeats):
        all_events.append((time, "ダウンビート", i+1))
    
    # 通常のビート（ダウンビート以外）を追加
    for time in beats:
        if time not in downbeats:  # ダウンビートでないビートのみ追加
            # 該当する小節番号を計算
            bar_number = 0
            for i, db_time in enumerate(downbeats):
                if time > db_time:
                    bar_number = i + 1
                else:
                    break
            
            all_events.append((time, "ビート", bar_number))
    
    # 時間順にソート
    all_events.sort(key=lambda x: x[0])
    
    print("\n=== 時系列順のイベント ===")
    for time, event_type, bar_number in all_events:
        bar_str = f"{bar_number:03d}" if bar_number > 0 else "000"
        
        # 時間を四捨五入して小数点1桁にし、0埋め
        time_str = f"{time:.1f}".zfill(5)
        
        print(f"時間: {time_str} 秒, 小節: {bar_str}, タイプ: {event_type}")
