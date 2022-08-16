from torch import nn
import torch
import torch.nn.functional as F


from modules.transformer.pose_tokenpose_b import get_pose_net
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
# from torch.nn import BatchNorm2d


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, cfg,  estimate_jacobian=False)
        super(KPDetector, self).__init__()

        self.predictor = get_pose_net(cfg, is_train, **kwargs)

        self.num_kp = num_kp

        if estimate_jacobian:
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None
        # self.temperature = temperature

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3)) # N * 10 * 2
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap
        out['feature_map'] = feature_map

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            # N * 10 *2 *2
            out['jacobian'] = jacobian


        return out
