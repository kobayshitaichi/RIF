import cv2


def opt(start_frame, end_frame, interval_frames, file_path):
    # 動画ファイルのロード
    video = cv2.VideoCapture(file_path)
    i = start_frame + interval_frames
    # 最初のフレームに移動して取得
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = video.read()

    # グレースケールにしてコーナ特徴点を抽出
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

    feature_params = {
        "maxCorners": 5,  # 特徴点の上限数
        "qualityLevel": 0.3,  # 閾値　（高いほど特徴点数は減る)
        "minDistance": 12,  # 特徴点間の距離 (近すぎる点は除外)
        "blockSize": 12,  #
    }
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # OpticalFlowのパラメータ
    lk_params = {
        "winSize": (15, 15),  # 特徴点の計算に使う周辺領域サイズ
        "maxLevel": 2,  # ピラミッド数 (デフォルト0で、2の場合は1/4の画像まで使われる)
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        ),  # 探索アルゴリズムの終了条件
    }

    for i in range(start_frame + interval_frames, end_frame + 1, interval_frames):
        try:
            result = True
            # 次のフレームを取得してグレースケールにする
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # OpticalFlowの計算
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, p0, None, **lk_params
            )

            # フレーム前後でトラックが成功した特徴点のみを

            identical_p1 = p1[status == 1]
            identical_p0 = p0[status == 1]
            prev_gray = frame_gray.copy()
            p0 = identical_p1.reshape(-1, 1, 2)
        except:
            result = False
            break

    return result
