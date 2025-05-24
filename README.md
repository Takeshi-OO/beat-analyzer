# marria用のbeat検知テストプロジェクト

各ファイルは連携しておらず個別で動作する

### detect_beats_librosa.py
非Deep LarningのLibrosaにてビートとビート外アタックのオンセットを検知。ダウンビートが欲しいため不採用


### downbeats_strong_onsets.py
Deep larningのMadmomを使用してダウンビート、ビート、ビート外アタックとそのオンセット強度を検知し、なおかつオンセットは強度順にダウンビート数と同数に絞った。great.wavでテストすると、ビートとダウンビートは正確だが、オンセット強度が不正確だったので不採用。


### exportBeats.py
downbeats_strong_onsets.pyのダウンビートとビート検出の機能だけ抜き出して、json出力機能を追加したもの。


### 結論
MVPではexportBeats.pyを採用する。
使用方法: python exportBeats.py audio/<音声ファイル> <出力JSONファイル>
ただし、強度の高いビートやアタックを優先する機能は将来追加したいため、他の二つも削除せず残しておく