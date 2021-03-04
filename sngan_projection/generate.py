import os

import numpy as np
import chainer
import yaml
import imageio

from sngan_projection import yaml_utils
from sngan_projection.miscs.downloader import download_weights


PATH = os.path.dirname(__file__)


class Generator(object):

    def __init__(self, weights_path=None, gpu=None, seed=None):
        if seed is not None:
            os.environ['CHAINER_SEED'] = f'{seed}'
            np.random.seed(seed)

        self.gen = model(weights_path)
        self.use_gpu = gpu is not None
        if self.use_gpu:
            chainer.cuda.get_device_from_id(gpu).use()
            self.gen.to_gpu(gpu)

    def __call__(self, y=None, output_path=None, **kwargs):
        if y is None:
            y = [np.random.choice(self.gen.n_classes)]

        y = self.gen.xp.asarray(y, dtype=self.gen.xp.float32)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = self.gen(y=y, **kwargs).data

        if self.use_gpu:
            x = chainer.cuda.to_cpu(x)

        ims = np.asarray(np.clip((x + 1) * 127.5, 0, 255), dtype=np.uint8)
        ims = ims.transpose((0, 2, 3, 1))

        if output_path is not None and len(ims) == 1:
            imageio.save(output_path, ims[0])
        return ims


def model(weights_path=None):
    path = os.path.join(PATH, '../resources/sn_projection.yml')
    config = yaml_utils.Config(yaml.load(open(path), Loader=yaml.FullLoader))
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(
        gen_conf['fn'], gen_conf['name'], gen_conf['args'])

    if weights_path is None:
        weights_path = './weights.npy'
        download_weights('1TDGXDM4s_xJdHCDzXt18aqODpbWKk8qe', weights_path)
    chainer.serializers.load_npz(weights_path, gen)

    return gen


if __name__ == '__main__':
    Generator()(output_path='im.png')
