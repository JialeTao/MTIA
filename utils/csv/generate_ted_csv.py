import pandas as pd
import random
import os

id_csv = pd.read_csv('ted_id.csv')
id_csv = id_csv.to_dict()
id_list = list(id_csv['id'].values())
print(id_list)
ll = len(id_list)

out_file_name = './ted_large.csv'
# custom_driving = ['oNkBx4CZuEg#000000#001024.mp4', 'WlDYrq8K6nk#005943#006135.mp4']
selected_driving = ['0gks6ceq4eQ#014103#014264.mp4','FDhlOovaGrI#006410#006567.mp4','5aH2Ppjpcho#002059#002228.mp4','LcNvkhS4UYg#016989#017205.mp4',
                    'JKsHhXwqDqM#030363#030585.mp4','JKsHhXwqDqM#029401#029533.mp4','zbHe4RpgV80#003048#003221.mp4','zbHe4RpgV80#007350#007492.mp4',
                    'zpjxElfNpks#010561#010716.mp4','9zC2Bc22QfA#005225#005435.mp4','360bU-vBJOI#011030#011186.mp4','iB4MS1hsWXU#014911#015238.mp4',
                    'Jc9RdbHEFa0#010035#010186.mp4']
custom_csv = open(out_file_name, 'w')

random_src_id = random.sample(id_list, 20)

full_video = os.listdir('./TED384-png/test')
random_driving = []
random_source = []

for vid in id_list:
    # random source
    if vid in random_src_id:
        videos = [video for video in full_video if vid in video]
        random_source += random.sample(videos, 1)
    # random source

    # random driving
    vid = vid + '#'
    select_flag = 0
    for ide in selected_driving:
        if vid in ide:
            select_flag = 1
            break
    if select_flag == 1:
        continue
    else:
        videos = [video for video in full_video if vid in video]
        random_driving += random.sample(videos, 1)
# random_driving.remove(random_driving[-1])
random_driving = random.sample(random_driving, 27)
print("random driving len: %d" % len(random_driving))
print(random_driving)

custom_driving = selected_driving + random_driving
print("custom driving len: %d" % len(custom_driving))

custom_csv = {'source':[],'driving':[]}
for dri in custom_driving:
    custom_csv['source']+=random_source
    custom_csv['driving']+=[dri]*len(random_source)
# print(custom_csv)
df = pd.DataFrame(custom_csv)
# print(df)
df.to_csv(out_file_name, index=None)
