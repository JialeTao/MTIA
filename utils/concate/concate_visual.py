import imageio
import os
from tqdm import tqdm
from skimage.transform import resize
import glob
import numpy as np
custom_path  = 'mgif/0923_bs64-token192-re-jacob-token-light-kp-head-light-jacob-head-kpadam-beta0_5-no-data-preprocess-clip-kpgrad1/animation'
pca_path = '../articulated-animation/checkpoints/mgif256/animation'
# custom_path  = 'voxceleb1/0926_bs64-token192-repeat150-re-jacob-token-light-kp-head-light-jacob-head-kpadam-beta0_5-skips-no-data-preprocess-clip-kpgrad1/animation'
# pca_path = '../articulated-animation/checkpoints/vox256/animation'
# fomm_path = '../articulated-animation/checkpoints/1011_vox256-fomm-no-bg-no-syncbn/animation'
out_path = custom_path + '_cmp_pca'
if not os.path.exists(out_path):
    os.makedirs(out_path)
custom_path = sorted(glob.glob(os.path.join(custom_path, '*.mp4')))
pca_path = sorted(glob.glob(os.path.join(pca_path, '*.mp4')))
# fomm_path = sorted(glob.glob(os.path.join(fomm_path, '*.mp4')))
i = 0
for img_custom, img_pca in tqdm(zip(custom_path, pca_path)):
# for img_custom, img_pca, img_fomm in tqdm(zip(custom_path, pca_path, fomm_path)):
    reader_custom = imageio.get_reader(img_custom)
    reader_pca = imageio.get_reader(img_pca)
    # reader_fomm = imageio.get_reader(img_fomm)
    frames = []
    # try:
    # for frame_custom, frame_pca, frame_fomm in zip(reader_custom, reader_pca, reader_fomm):
    for frame_custom, frame_pca in zip(reader_custom, reader_pca):
        frame = np.concatenate((frame_custom[:,0:1024,:],frame_pca[:,768:1024,:],frame_custom[:,1536:1792,:]), axis=1)
        # frame = np.concatenate((frame_custom[:,0:512,:],frame_pca[:,768:1024,:],frame_custom[:,1024:1280,:]), axis=1)
        # frame = np.concatenate((frame_custom[:,0:512,:],frame_fomm[:,768:1024,:],frame_pca[:,768:1024,:],frame_custom[:,1024:1280,:]), axis=1)
        # print(frame.shape)
        frames.append(frame)
    # except:
    #    pass
    # print(len(frames))
    reader_custom.close()
    # reader_fomm.close()
    reader_pca.close()
    video = np.array(frames)
    # print(video.shape)
    imageio.mimsave(os.path.join(out_path, os.path.basename(img_custom)), video)
    # print(frames[0].shape)
    # os.makedirs(os.path.join(d_out, image))
    # [imageio.imsave(os.path.join(d_out, image, str(i).zfill(7) + '.png'), frame) for i, frame in enumerate(frames)]
