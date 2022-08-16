import pandas as pd


out_file_name = './data/mgif256_large.csv'
# custom_driving = ['oNkBx4CZuEg#000000#001024.mp4', 'WlDYrq8K6nk#005943#006135.mp4']
custom_driving = [9,14,16,18,23,24,37,39,40,47,
                  50,52,55,56,58,61,67,71,73,75,
                  76,77,78,83,86,87,98,99]
custom_csv = open(out_file_name, 'w')

custom_csv = {'source':[],'driving':[]}
for dri in custom_driving:
    for src in custom_driving:
        custom_csv['source'].append(str(src).zfill(5)+'.gif')
        custom_csv['driving'].append(str(dri).zfill(5)+'.gif')
print(custom_csv)
df = pd.DataFrame(custom_csv)
# print(df)
df.to_csv(out_file_name, index=None)
