from source.links.sn_linear import SNLinear
from source.links.sn_embed_id import SNEmbedID
import chainer
import chainer.links as L
from chainer import functions as F
from sngan_projection.model.resblocks import Block, DisBlock, OptimizedBlock
from sngan_projection.miscs.random_samples import sample_categorical, sample_continuous


class ResNetGenerator(chainer.Chain):

    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal"):
        super().__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 16, initialW=initializer)
            self.block2 = Block(ch * 16, ch * 16, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block5 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
            self.block6 = Block(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b7 = L.BatchNormalization(ch)
            self.l7 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, batchsize=64, z=None, y=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                                   xp=self.xp) if self.n_classes > 0 else None
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise ValueError('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.block6(h, y, **kwargs)
        h = self.b7(h)
        h = self.activation(h)
        h = F.tanh(self.l7(h))
        return h


class ResNetDiscriminator(chainer.Chain):

    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super().__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = DisBlock(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = DisBlock(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = DisBlock(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = DisBlock(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block6 = DisBlock(ch * 16, ch * 16, activation=activation, downsample=False)
            self.l7 = SNLinear(ch * 16, 1, initialW=initializer)
            if n_classes > 0:
                self.l_y = SNEmbedID(n_classes, ch * 16, initialW=initializer)

    def __call__(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l7(h)
        if y is not None:
            w_y = self.l_y(y)
            output += F.sum(w_y * h, axis=1, keepdims=True)
        return output
