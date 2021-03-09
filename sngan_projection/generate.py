import os

import numpy as np
import chainer

from sngan_projection.model import resnet
from sngan_projection.miscs.downloader import download_weights


class Discriminator(resnet.ResNetDiscriminator):

    def __init__(self, weights_path='weights_dis.npy', gpu=None, seed=None):
        if seed is not None:
            os.environ['CHAINER_SEED'] = f'{seed}'
            np.random.seed(seed)

        super().__init__(ch=64, n_classes=1000)

        filename = os.path.join(weights_path, 'weights_dis.npy')
        if not os.path.isfile(filename):
            download_weights('1k27RlxEJgjUzPfbIqusGAmHhEKKp59rs', filename)
        chainer.serializers.load_npz(filename, self)

        self.use_gpu = gpu is not None
        if self.use_gpu:
            chainer.cuda.get_device_from_id(gpu).use()
            self.to_gpu(gpu)

    def __call__(self, x, y=None):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = super().__call__(x, y=y).data

        if self.use_gpu:
            x = chainer.cuda.to_cpu(x)

        return x


class Generator(resnet.ResNetGenerator):

    def __init__(self, weights_path='./', gpu=None, seed=None):
        if seed is not None:
            os.environ['CHAINER_SEED'] = f'{seed}'
            np.random.seed(seed)

        super().__init__(ch=64, dim_z=128, bottom_width=4, n_classes=1000)

        filename = os.path.join(weights_path, 'weights_gen.npy')
        if not os.path.isfile(filename):
            download_weights('1TDGXDM4s_xJdHCDzXt18aqODpbWKk8qe', filename)
        chainer.serializers.load_npz(filename, self)

        self.use_gpu = gpu is not None
        if self.use_gpu:
            chainer.cuda.get_device_from_id(gpu).use()
            self.to_gpu(gpu)

        self.dis = Discriminator(weights_path=weights_path, gpu=gpu, seed=seed)

    def __call__(self, batchsize=64, z=None, y=None, **kwargs):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = super().__call__(batchsize=batchsize, z=z, y=y, **kwargs)
            is_real = self.dis(x)

        if self.use_gpu:
            x = chainer.cuda.to_cpu(x.data)

        ims = np.asarray(np.clip((x + 1) * 127.5, 0, 255), dtype=np.uint8)
        ims = ims.transpose((0, 2, 3, 1))

        return ims, is_real


if __name__ == '__main__':
    gen = Generator()
    ims, is_real = gen(y=[gen.xp.random.randint(gen.n_classes)])
    import imageio
    imageio.save('im.png', ims[0])
