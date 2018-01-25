from mxnet import gluon
from mxnet.initializer import Xavier


class E2FAR(gluon.HybridBlock):
    def __init__(self, freeze=False, batch_norm=False, **kwargs):
        super(E2FAR, self).__init__(**kwargs)
        with self.name_scope():
            self.layers = [2, 2, 3, 3]
            self.filters = [64, 128, 256, 512]
            self.hidden_units = [4096, 1024]
            self.backbone = self._make_features([2, 2, 3, 3], [64, 128, 256, 512], batch_norm)
            self.extra_backbone = self._make_features([3], [512], batch_norm)
            self.conv6 = gluon.nn.Conv2D(512, kernel_size=5, strides=2, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros')
            self.conv7 = gluon.nn.Conv2D(512, kernel_size=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros')
            self.conv8 = gluon.nn.Conv2D(512, kernel_size=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros')
            self.shape_regressor = self._make_prediction(out_dim=199)
            self.exp_regressor = self._make_prediction(out_dim=29)

        if freeze:
            for _, w in self.backbone.collect_params().items():
                w.grad_req = 'null'

            for _, w in self.extra_backbone.collect_params().items():
                w.grad_req = 'null'

    def _make_features(self, layers, filters, batch_norm):
        featurizer = gluon.nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(gluon.nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                               weight_initializer=Xavier(rnd_type='gaussian',
                                                                         factor_type='out',
                                                                         magnitude=2),
                                               bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(gluon.nn.BatchNorm())
                featurizer.add(gluon.nn.Activation('relu'))
            featurizer.add(gluon.nn.MaxPool2D(strides=2))
        return featurizer

    def _make_prediction(self, out_dim):
        preds = gluon.nn.HybridSequential(prefix='')
        for units in self.hidden_units:
            preds.add(gluon.nn.Dense(units))

        preds.add(gluon.nn.Dense(out_dim))

        return preds

    def hybrid_forward(self, F, x):
        feat1 = self.backbone(x)
        feat2 = self.extra_backbone(feat1)
        output_shape = self.shape_regressor(feat2)

        feat1_ = self.conv6(feat1)
        feat2_ = self.conv7(feat2)
        feat_fused = F.concat(feat1_, feat2_, dim=1)
        feat_fused = self.conv8(feat_fused)

        output_exp = self.exp_regressor(feat_fused)

        return output_shape, output_exp
