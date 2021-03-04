import os

import numpy as np
import chainer

from sngan_projection.model.resnet import ResNetGenerator
from sngan_projection.miscs.downloader import download_weights


class Generator(ResNetGenerator):

    def __init__(self, weights_path=None, gpu=None, seed=None):
        if seed is not None:
            os.environ['CHAINER_SEED'] = f'{seed}'
            np.random.seed(seed)

        super().__init__(ch=64, dim_z=128, bottom_width=4, n_classes=1000)

        if weights_path is None:
            weights_path = './weights.npy'
            download_weights('1TDGXDM4s_xJdHCDzXt18aqODpbWKk8qe', weights_path)
        chainer.serializers.load_npz(weights_path, self)

        self.use_gpu = gpu is not None
        if self.use_gpu:
            chainer.cuda.get_device_from_id(gpu).use()
            self.to_gpu(gpu)

    def __call__(self, batchsize=64, z=None, y=None, **kwargs):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = self.gen(batchsize=batchsize, z=z, y=y, **kwargs).data

        if self.use_gpu:
            x = chainer.cuda.to_cpu(x)

        ims = np.asarray(np.clip((x + 1) * 127.5, 0, 255), dtype=np.uint8)
        ims = ims.transpose((0, 2, 3, 1))

        return ims


if __name__ == '__main__':
    gen = Generator()
    ims = gen(y=[gen.xp.random.randint(gen.n_classes)])
    import imageio
    imageio.save('im.png', ims[0])
