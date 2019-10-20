import os, sys, time, shutil, requests

import numpy as np
import tqdm
import chainer
import yaml
from PIL import Image

from sngan_projection import yaml_utils
from sngan_projection.miscs.downloader import download_weights


PATH = os.path.dirname(__file__)


class Generate(object):

    def __init__(self,
                 weights_path=os.path.join(PATH, '../resources/ResNetGenerator_850000.npz'),
                 categories=None,
                 seed=None):
        if seed is not None:
            os.environ['CHAINER_SEED'] = f'{seed}'
            np.random.seed(seed)

        path = os.path.join(PATH, '../resources/sn_projection.yml')
        config = yaml_utils.Config(yaml.load(open(path), Loader=yaml.FullLoader))
        gen_conf = config.models['generator']
        self.gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
        
        if not os.path.isfile(weights_path):
            download_weights('1TDGXDM4s_xJdHCDzXt18aqODpbWKk8qe', weights_path)
        chainer.serializers.load_npz(weights_path, self.gen)

        if categories is None:
            path = os.path.join(PATH, '../resources/synset_words.txt')
            with open(path) as f:
                categories = [l.strip('\n').split() for l in f.readlines() if l[0] != '#']
        self.categories = categories

    def __call__(self, output_path='im.png'):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            sel = np.random.choice(self.categories, size=2, replace=False)
            nums = [int(s[0]) for s in sel]
            c_vec = [1 / len(nums) if i in nums else 0 for i in range(self.gen.n_classes)]
            y = self.gen.xp.asarray([c_vec], dtype=self.gen.xp.float32)
            x = self.gen(1, y=y).data
            im = np.asarray(np.clip(x * 127.5 + 127.5, 0, 255), dtype=np.uint8)[0]
            im = im.transpose((1, 2, 0))
            im = Image.fromarray(im)
            im.save(output_path)


if __name__ == '__main__':
    Generator()()