import os

import numpy as np
import chainer
import yaml
from PIL import Image

from sngan_projection import yaml_utils
from sngan_projection.miscs.downloader import download_weights


PATH = os.path.dirname(__file__)


class Generator(object):

    def __init__(self, weights_path=None, seed=None):
        if seed is not None:
            os.environ['CHAINER_SEED'] = f'{seed}'
            np.random.seed(seed)

        self.gen = model(weights_path)

    def __call__(self, categories=None, output_path=None):
        if categories is None:
            categories = [np.random.choice(self.gen.n_classes)]

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            c_vec = np.zeros(self.gen.n_classes)
            c_vec[np.array(categories)] = 1 / len(categories)
            y = self.gen.xp.asarray([c_vec.tolist()], dtype=self.gen.xp.float32)
            x = self.gen(1, y=y).data

        im = np.asarray(np.clip(x * 127.5 + 127.5, 0, 255), dtype=np.uint8)[0]
        im = im.transpose((1, 2, 0))

        if output_path is not None:
            im = Image.fromarray(im)
            im.save(output_path)
        else:
            return im


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
