import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, bg_predictor=None):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector, bg_predictor=bg_predictor)
    else:
        print('warining: reconstruction without checkpoiont, make sure you are using the trained models...')
        # raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    if bg_predictor is not None:
        bg_predictor.eval()

    generator.eval()
    kp_detector.eval()

    from frames_dataset import read_video
    for it, x in tqdm(enumerate(dataloader)):

        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)

                if bg_predictor is not None:
                    bg_params = bg_predictor(source,driving)
                else:
                    bg_params = None

                out = generator(source, kp_source=kp_source, kp_driving=kp_driving, bg_params=bg_params)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                # out['kp_norm'] = kp_driving
                del out['sparse_deformed']
                del out['deformed']
                # del out['occlusion_map']
                # del out['prediction']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out,animate=False)
                visualizations.append(visualization)
                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    # print(len(loss_list))
    print("Reconstruction loss: %s" % np.mean(loss_list))
    return loss_list