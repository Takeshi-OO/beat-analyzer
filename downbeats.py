import sys
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

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

if __name__ == "__main__":
    audio_file = "audio/bgm1.wav"  # 解析したいオーディオファイルへのパス
    results = detect_downbeats(audio_file)
    
    # 結果を出力
    # 例: time=1.23, type=1 -> 1.23秒付近にダウンビート
    for (beat_time, beat_type) in results:
        beat_label = "Downbeat" if beat_type == 1 else "Beat"
        print(f"Time: {beat_time:.2f} sec, Type: {beat_label}")
