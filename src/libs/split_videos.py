import ffmpeg
import pandas as pd
import os
from tqdm import tqdm
import argparse


def split(video_name, root_dir):
    # 動画をフレーム毎の静止画に分割
    print("{}: Convert to Images start".format(video_name))
    path = root_dir + "/videos/" + video_name + "/" + video_name + ".mp4"
    stream = ffmpeg.input(path)
    stream = ffmpeg.output(
        stream,
        root_dir + "/dataset/test/" + video_name + "/images/00000" + "_%7d_on" + ".jpg",
        r=30,
    )
    ffmpeg.run(stream)


def split_test_data(video_name, root_dir):
    # アノテーション結果をDataframe型にする
    path = root_dir + "/videos/" + video_name + "/" + video_name + ".txt"
    df = pd.read_table(path, header=None)
    df.columns = columns = ["a", "b", "c", "start", "e", "end", "g", "length", "onoff"]
    df = df.drop(["a", "b", "c", "e", "g", "end"], axis=1)

    # 手術動画を分割
    print("{}: Split videos start".format(video_name))
    path = root_dir + "/videos/" + video_name + "/" + video_name + ".mp4"
    for i in tqdm(range(len(df))):
        s = str(i)
        stream = ffmpeg.input(path, ss=df["start"][i], t=df["length"][i])
        stream = ffmpeg.output(
            stream,
            root_dir
            + "/videos/"
            + video_name
            + "/split_videos/videos_{}.mp4".format(s.zfill(3)),
        )
        ffmpeg.run(stream)

    # 動画をフレーム毎の静止画に分割
    print("{}: Convert to Images start".format(video_name))
    for i in tqdm(range(len(df))):
        s = str(i)
        path = root_dir + "/videos/" + video_name + "/split_videos/videos_"
        path = path + s.zfill(3) + ".mp4"
        stream = ffmpeg.input(path)
        stream = ffmpeg.output(
            stream,
            root_dir
            + "/dataset/test/"
            + video_name
            + "/images/"
            + s.zfill(5)
            + "_%7d_"
            + df["onoff"][i]
            + ".jpg",
            r=30,
        )
        ffmpeg.run(stream)

    # 分割動画を削除
    # shutil.rmtree('./TestImages/videos')


def split_train_data(video_name, root_dir):
    # アノテーション結果をDataframe型にする
    path = root_dir + "/videos/" + video_name + "/" + video_name + ".txt"
    df = pd.read_table(path, header=None)
    df.columns = columns = ["a", "b", "c", "start", "e", "end", "g", "length", "onoff"]
    df = df.drop(["a", "b", "c", "e", "g", "end"], axis=1)

    # #手術動画を分割
    if len(os.listdir("../videos/" + video_name + "/split_videos")) == 0:
        print("{}: Split videos start".format(video_name))
        path = root_dir + "/videos/" + video_name + "/" + video_name + ".mp4"
        for i in tqdm(range(len(df))):
            s = str(i)
            stream = ffmpeg.input(path, ss=df["start"][i], t=df["length"][i])
            stream = ffmpeg.output(
                stream,
                "../videos/"
                + video_name
                + "/split_videos/videos_{}.mp4".format(s.zfill(3)),
            )
            ffmpeg.run(stream)

    # 動画をフレーム毎の静止画に分割
    print("{}: Convert to images start".format(video_name))
    for i in tqdm(range(len(df))):
        s = str(i)
        path = root_dir + "/videos/" + video_name + "/split_videos/videos_"
        path = path + s.zfill(3) + ".mp4"
        stream = ffmpeg.input(path, f="mp4")
        output_path = (
            root_dir
            + "/dataset/train/images/"
            + df["onoff"][i]
            + "/"
            + video_name
            + "_"
            + s.zfill(3)
            + "_"
            + df["onoff"][i]
            + "_%10d.jpg"
        )
        stream = ffmpeg.output(stream, output_path, r=5)
        ffmpeg.run(stream)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',help='train or test')
    parser.add_argument('-name')
    args = parser.parse_args()
    data = args.data
    name = args.name

    if data == 'test':
        split_test_data(name)
    elif data == 'train':
        split_train_data(name)
    elif data == 'split':
        split(name)

    else:
        print('-data train or test')