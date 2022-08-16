import os

data_dir = './data/vox-png/train'
data_list = []
for data in os.listdir(data_dir):
    data_list.append(data.split('#')[0])
id_list = set(data_list)
print(len(id_list))

data_dir = './data/voxceleb1-png/train'
data_list = []
for data in os.listdir(data_dir):
    data_list.append(data.split('#')[0])
id_list_celeb1  = set(data_list)
print(len(id_list_celeb1))

print('id in vox but not in voxceleb1:')
for id in id_list:
    if id not in id_list_celeb1:
        print(id)

print('id in voxceleb1 but not in vox:')
for id in id_list_celeb1:
    if id not in id_list:
        print(id)

