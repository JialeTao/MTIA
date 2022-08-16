import os
import pandas as pd
import random


full_video = os.listdir('./fashion-png/test')
random_driving = random.sample(full_video, 50)
random_source = random.sample(full_video, 20)
ll = len(random_source)

out_file_name = './fashion_random_dri_src_large.csv'

custom_csv = {'source':[],'driving':[]}
for dri in random_driving:
    custom_csv['source']+=random_source
    custom_csv['driving']+=[dri]*ll
# print(custom_csv)
df = pd.DataFrame(custom_csv)
# print(df)
df.to_csv(out_file_name, index=None)
