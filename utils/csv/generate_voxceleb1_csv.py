import os
import pandas as pd
import random

src_csv = pd.read_csv('voxceleb1_mid_left_right_src.csv')
src_csv = src_csv.to_dict()
source = list(src_csv['source'].values())
source = list(set(source))
ll = len(source)
print(source, ll)

video_id = [video.split('#')[0] for video in source]
video_id = sorted(list(set(video_id)))
video_id.remove('id10306')
print(video_id)

full_video = os.listdir('./voxceleb1-png/test')
random_driving = []
for vid in video_id:
    vid = vid + '#'
    videos = [video for video in full_video if vid in video]
    random_driving += random.sample(videos, 3)
random_driving.remove(random_driving[-1])
random_driving.remove(random_driving[-3])
random_driving.remove(random_driving[-5])
random_driving.remove(random_driving[-7])
print("random driving len: %d" % len(random_driving))
print(random_driving)

out_file_name = './voxceleb1_mlr_src_random_dri_large.csv'
# custom_driving = ['oNkBx4CZuEg#000000#001024.mp4', 'WlDYrq8K6nk#005943#006135.mp4']
custom_csv = open(out_file_name, 'w')

custom_csv = {'source':[],'driving':[]}
for dri in random_driving:
    custom_csv['source']+=source
    custom_csv['driving']+=[dri]*ll
# print(custom_csv)
df = pd.DataFrame(custom_csv)
# print(df)
df.to_csv(out_file_name, index=None)
