"""
Implementation of End-to-end 3D Face Reconstruction with Deep Neural Network
"""
import argparse
import scipy.io as sio
import cv2
import pandas as pd
import numpy as np
import math
import mxnet as mx
from mxnet import nd, autograd, gluon

from utils import *
from model import E2FAR


def enlarge_bbox(x, y, w, h, enlarge_factor=1.2):
    x = x - (enlarge_factor - 1) * w
    y = y - (enlarge_factor - 1) * h

    w = w * (2 * enlarge_factor - 1)
    h = h * (2 * enlarge_factor - 1)

    return int(x), int(y), int(w), int(h)


class SupervisedDataset(gluon.data.Dataset):
    def __init__(self, file_path, is_train=True, img_size=180, enlarge_factor=0.9):
        self.data_frame = pd.read_csv(file_path)
        self.is_train = is_train
        self.img_size = img_size
        self.enlarge_factor = enlarge_factor

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x, y, w, h = self.data_frame.iloc[idx, 1:5]
        l, t, ww, hh = enlarge_bbox(x, y, w, h, self.enlarge_factor)
        r, b = l + ww, t + hh

        img = img[t: b, l:r, :]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) - 127.5

        img = nd.transpose(nd.array(img), (2, 0, 1))

        label_path = img_path.replace('.jpg', '.mat')

        label = sio.loadmat(label_path)

        params_shape = label['Shape_Para'].astype(np.float32).ravel()
        params_exp = label['Exp_Para'].astype(np.float32).ravel()

        return img, params_shape, params_exp


def multi_factor_scheduler(lr_step_epochs, num_samples, batch_size, lr_factor, start_epoch):
    num_samples = num_samples
    epoch_size = int(math.ceil(float(num_samples) / batch_size))
    step_epochs = [int(l) - start_epoch for l in lr_step_epochs.split(',')]
    steps = [epoch_size * x for x in step_epochs]

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(steps, factor=lr_factor)

    return lr_scheduler


def initialize_inference(inference, pretrained, start_epoch):
    if pretrained:
        print('Loading the pretrained model')
        vggface_weights = nd.load('ckpt/VGG-FACE/VGG_FACE-0000.params')
        # change the name
        checkpoint = {}
        vgg_face_layers = [2, 2, 3, 3, 3]
        for k, v in vggface_weights.items():
            if 'conv' in k:
                ind1, ind2, sub_name = k.split('_')
                ind1 = int(ind1.replace('arg:conv', '')) - 1
                ind2 = int(ind2[-1]) - 1
                ind = sum(vgg_face_layers[:ind1]) + ind2
                key = inference.name + '_conv' + str(ind) + '_' + sub_name
                checkpoint[key] = v

        # load the weights
        for k in inference.collect_params().keys():
            if k in checkpoint:
                inference.collect_params()[k]._load_init(checkpoint[k], ctx)
                print('Loaded %s weights from checkpoints' % k)
            else:
                inference.collect_params()[k].initialize(ctx=ctx)
                print('Initialize %s weights' % k)
        print('Done')
    elif start_epoch > 0:
        print('Loading the weights from [%d] epoch' % start_epoch)
        inference.load_params(os.path.join(args.ckpt_dir, args.prefix, '%s-%d.params' % (args.prefix, start_epoch)), ctx)
    else:
        inference.collect_params().initialize(ctx=ctx)
    return inference


class ProjectionL2Loss(gluon.loss.Loss):
    def __init__(self, weights, weight=1, batch_axis=0, **kwargs):
        super(ProjectionL2Loss, self).__init__(weight, batch_axis, **kwargs)
        self._weights = weights
        self._batch_axis = batch_axis

    def hybrid_forward(self, F, pred, label):
        pred = F.dot(pred, self._weights)
        label = F.dot(label, self._weights)
        loss = F.square(pred - label)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


def train():
    print('Start to train')
    logger = add_logger(args.log_dir, args.prefix, remove_previous_log=args.start_epoch == 0)
    check_ckpt(args.ckpt_dir, args.prefix)
    model3d = sio.loadmat(args.model3d)
    shape_pc = nd.array(model3d['shapePC']).transpose().as_in_context(ctx)
    exp_pc = nd.array(model3d['expPC']).transpose().as_in_context(ctx)

    trainset = SupervisedDataset(args.train_list)
    valset = SupervisedDataset(args.val_list)

    train_loader = gluon.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = gluon.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    inference = E2FAR(freeze=args.freeze)
    # initialize
    initialize_inference(inference, args.pretrained, args.start_epoch)

    lr_scheduler = multi_factor_scheduler(args.lr_steps, len(trainset), args.batch_size, args.lr_factor, args.start_epoch)
    trainer = gluon.Trainer(inference.collect_params(), optimizer='adam',
                            optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'lr_scheduler': lr_scheduler})

    criterion_shape = ProjectionL2Loss(shape_pc)
    criterion_exp = ProjectionL2Loss(exp_pc)
    metric_shape, metric_exp = mx.metric.Loss('shape-loss'), mx.metric.Loss('exp-loss')

    for cur_epoch in range(args.start_epoch + 1, args.epochs + 1):
        metric_shape.reset()
        metric_exp.reset()

        for i, batch in enumerate(train_loader):
            data = batch[0].as_in_context(ctx)
            gt_shape = batch[1].as_in_context(ctx)
            gt_exp = batch[2].as_in_context(ctx)

            with autograd.record():
                preds_shape, preds_exp = inference(data)
                loss_shape = criterion_shape(preds_shape, gt_shape)
                loss_exp = criterion_exp(preds_exp, gt_exp)

                loss = loss_shape + 5 * loss_exp
                loss.backward()

            trainer.step(data.shape[0])

            metric_shape.update(None, preds=loss_shape)
            metric_exp.update(None, preds=loss_exp)

            if i % args.log_interval == 0 and i > 0:
                logger.info('Epoch [%d] Batch [%d]: shape loss=%f, exp loss=%f, total loss=%f' %
                            (cur_epoch, i, metric_shape.get()[1], metric_exp.get()[1],
                             0.001 * metric_shape.get()[1] + metric_exp.get()[1]))

        logger.info('Epoch [%d]: train-shape-loss=%f' % (cur_epoch, metric_shape.get()[1]))
        logger.info('Epoch [%d]: train-exp-loss=%f' % (cur_epoch, metric_exp.get()[1]))

        inference.save_params(os.path.join(args.ckpt_dir, args.prefix, '%s-%d.params' % (args.prefix, cur_epoch)))

        metric_shape.reset()
        metric_exp.reset()

        for i, batch in enumerate(val_loader):
            data = batch[0].as_in_context(ctx)
            gt_shape = batch[1].as_in_context(ctx)
            gt_exp = batch[2].as_in_context(ctx)

            preds_shape, preds_exp = inference(data)

            loss_shape = criterion_shape(preds_shape, gt_shape)
            loss_exp = criterion_exp(preds_exp, gt_exp)

            metric_shape.update(None, preds=loss_shape)
            metric_exp.update(None, preds=loss_exp)

        logger.info('Epoch [%d]: val-shape-loss=%f' % (cur_epoch, metric_shape.get()[1]))
        logger.info('Epoch [%d]: val-exp-loss=%f' % (cur_epoch, metric_exp.get()[1]))

    print('Done')


def test():

    testset = SupervisedDataset(args.test_list)

    test_loader = gluon.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    inference = E2FAR(freeze=args.freeze)
    # initialize
    initialize_inference(inference, args.pretrained, args.start_epoch)

    for i, batch in enumerate(test_loader):
        data = batch[0].as_in_context(ctx)
        preds_shape, preds_exp = inference(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # log
    parser.add_argument('--ckpt_dir', default='ckpt', help='checkpoint directory')
    parser.add_argument('--log_dir', default='logs', help='log directory')
    parser.add_argument('--prefix', default='E2FAR', type=str, help='prefix')
    parser.add_argument('--arch', default='vgg16', type=str, help='base architecture')
    # train
    parser.add_argument('--model3d', default='', type=str, help='3D model')
    parser.add_argument('--train_list', default='', type=str, help='wider train record')
    parser.add_argument('--gpu', default=0, type=int, help='gpus')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--epochs', default=100, type=int, help='total epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--pretrained', action='store_true', help='pre-trained model')
    parser.add_argument('--freeze', action='store_true', help='freeze parameters')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--lr_steps', default='80, 160', help='learning decay steps')
    parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decrease factor')
    parser.add_argument('--log_interval', default=20, type=int, help='log interval')
    parser.add_argument('--training', dest='training', action='store_true', help='training flag')
    # validate
    parser.add_argument('--val_list', default='', type=str, help='validation record')
    # test
    parser.add_argument('--testing', dest='training', action='store_false', help='testing flag')
    parser.add_argument('--test_list', default='', type=str, help='test record')

    parser.set_defaults(training=True)

    args = parser.parse_args()

    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    data_shape = tuple(map(int, args.data_shape.split(',')))

    if args.training:
        train()
    else:
        test()
