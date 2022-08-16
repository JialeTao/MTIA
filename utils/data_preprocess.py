import imageio
import os
from tqdm import tqdm
from skimage.transform import resize

d = 'data/moving-gif-png/test'
d_out = 'data/moving-gif-png/test-png'
os.makedirs(d_out)
for image in tqdm(os.listdir(d)):
    try:
       reader = imageio.get_reader(os.path.join(d, image))
       frames = []
       for frame in reader:
           # frames.append(resize(frame, (256, 256)))
           frames.append(frame)
       reader.close()
    except:
       reader.close()
    os.makedirs(os.path.join(d_out, image))
    [imageio.imsave(os.path.join(d_out, image, str(i).zfill(5) + '.png'), frame) for i, frame in enumerate(frames)]
