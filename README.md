# mxnet-E2FAR
This is a MXNet/Gluon Implementation of End-to-end 3D Face Reconstruction with Deep Neural Networks.

1. Download VGG-Face and convert it to the mxnet-weights by running the caffe_converter:
    ```
    python $MXNET/tools/caffe_converter/convert_model.py prototxt weights params_name
    ```
Put the weights into the folder ```ckpt/VGG-Face```

2. Prepare the dataset

3. For train your dataset, you may need to change the ```dataset``` in the main code to fit your dataset

4. Run the code:
    ```
    # fine-tune the branch and fully connected layers
    python E2FAR.py --pretrained --freeze --epoch 10

    # fine-tune whole network
    python E2FAR.py --start_epoch 10
    ```

If you use this code, pls mention this repo and cite the paper:
```
@InProceedings{Dou_2017_CVPR,
author = {Dou, Pengfei and Shah, Shishir K. and Kakadiaris, Ioannis A.},
title = {End-To-End 3D Face Reconstruction With Deep Neural Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

## Known issues
dataloader is very slow and cannot make fully usage of GPU training.
You can use record io to pack the image and do more augmentation.