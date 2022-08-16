import imageio
import os
from tqdm import tqdm
from skimage.transform import resize
import glob
import numpy as np
custom_path  = 'voxceleb1/0926_bs64-token192-repeat150-re-jacob-token-light-kp-head-light-jacob-head-kpadam-beta0_5-skips-no-data-preprocess-clip-kpgrad1/reconstruction-mask-kp'
pca_path = '../articulated-animation/checkpoints/vox256/reconstruction-mask'
fomm_path = '../articulated-animation/checkpoints/1011_vox256-fomm-no-bg-no-syncbn/reconstruction/'
fomm_reg_path = '../articulated-animation/checkpoints/1005_vox256-fomm-regression-repeat150-no-bg-with-syncbn/reconstruction-mask/'
out_path = custom_path + 'cmp-reg-fomm-pca'
if not os.path.exists(out_path):
    os.makedirs(out_path)
custom_path = sorted(glob.glob(os.path.join(custom_path, '*.mp4')))
pca_path = sorted(glob.glob(os.path.join(pca_path, '*.mp4')))
fomm_path = sorted(glob.glob(os.path.join(fomm_path, '*.mp4')))
fomm_reg_path = sorted(glob.glob(os.path.join(fomm_reg_path, '*.mp4')))
i = 0
for img_custom, img_pca, img_fomm, img_fomm_reg in tqdm(zip(custom_path, pca_path, fomm_path,fomm_reg_path)):
# for img_custom, img_pca in tqdm(zip(custom_path, pca_path)):
    reader_custom = imageio.get_reader(img_custom)
    reader_pca = imageio.get_reader(img_pca)
    reader_fomm = imageio.get_reader(img_fomm)
    reader_fomm_reg= imageio.get_reader(img_fomm_reg)
    frames = []
    # try:
    for frame_custom, frame_pca, frame_fomm, frame_fomm_reg in zip(reader_custom, reader_pca, reader_fomm, reader_fomm_reg):
        # frame = np.concatenate((frame_custom[:,0:512,:],frame_pca[:,768:1024,:],frame_custom[:,1024:1280,:]), axis=1)
        frame_fomm = np.concatenate((frame_fomm[:,0:512,:],frame_fomm[:,768:1024,:],frame_fomm[:,1024:1280,:],frame_fomm[:,1792:,:]), axis=1)
        frame_fomm_reg = np.concatenate((frame_fomm_reg[:,0:512,:],frame_fomm_reg[:,768:1280,:],frame_fomm_reg[:,1792:,:]), axis=1)
        frame_pca = np.concatenate((frame_pca[:,0:512,:],frame_pca[:,768:4096,:]), axis=1)
        # print(frame.shape)
        frame = np.concatenate((frame_fomm_reg,frame_fomm,frame_pca,frame_custom), axis=0)
        frames.append(frame)
    # except:
    #    pass
    # print(len(frames))
    reader_custom.close()
    reader_pca.close()
    reader_fomm.close()
    reader_fomm_reg.close()
    video = np.array(frames)
    # print(video.shape)
    imageio.mimsave(os.path.join(out_path, os.path.basename(img_custom)), video)
    # print(frames[0].shape)
    # os.makedirs(os.path.join(d_out, image))
    # [imageio.imsave(os.path.join(d_out, image, str(i).zfill(7) + '.png'), frame) for i, frame in enumerate(frames)]
