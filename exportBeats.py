import sys
import json
import numpy as np
from madmom.audio import Signal
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

def export_beats_to_json(audio_file, output_file=None, fps=100, beat_per_bar=4):

    # 音声ファイルを読み込み
    signal = Signal(audio_file)
    sample_rate = signal.sample_rate
    
    # ビートとダウンビートを検出
    act = RNNDownBeatProcessor()(audio_file)
    beats_with_positions = DBNDownBeatTrackingProcessor(
        beats_per_bar=beat_per_bar, fps=fps)(act)
    
    # ビートとダウンビートを分離
    beats = beats_with_positions[:, 0]  # 時間情報
    positions = beats_with_positions[:, 1]  # 拍の位置情報（1が最初の拍）
    
    # JSON形式のデータを作成
    beat_data = []
    current_measure = 0
    
    # 各ビートを処理
    for i in range(len(beats)):
        beat_time = beats[i]
        position = positions[i]
        
        # 最初の拍（ダウンビート）なら小節を増やす
        if position == 1.0:
            current_measure += 1
            beat_in_measure = 1
        else:
            beat_in_measure = int(position)
        
        # 最初のダウンビート前のビートは measure=0 として扱う
        if current_measure == 0:
            beat_in_measure = int(position)
        
        beat_data.append({
            "time": round(float(beat_time), 2),
            "measure": current_measure,
            "beatInMeasure": beat_in_measure
        })
    
    # 結果をJSON形式で出力
    result = {
        "sampleRate": sample_rate,
        "beats": beat_data
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python downbeats.py <音声ファイル> [出力JSONファイル]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    export_beats_to_json(audio_file, output_file)
