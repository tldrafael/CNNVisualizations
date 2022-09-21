import os
import sys
from torch.utils.data import DataLoader
from gradcam import GradCAMModel, CAMImage
import datapipe as dp
import utils as ut
import training as tr


if __name__ == '__main__':
    void_classes = ut.void_classes['RTK']
    ds_params = {
        'n_classes': 13,
        'void_classes': ut.void_classes['RTK'],
    }

    tr_params = {
        'fl_maxpool': False,
        'fl_richstem': False,
        'fl_parallelstem': False,
        'fl_stemstride': True,
        'fl_transpose': False,
        'output_stride': 16,
    }

    modelpath = '/home/rafael/Downloads/CE.OS16.crop_cutmix80.woMaxPool/DeepLabV3Plus-resnet50/exp_0/model.best.pth'
    model = tr.load_model(modelpath, tr_params, ds_params, use_cpu=True)
    RTKdir = '/home/rafael/Workspace/Datasets/RTK_dataset/RTK_pisss'
    val_ds = dp.DatasetWithRelabel(annotation_file=os.path.join(RTKdir, 'labeled_list_classic_val.txt'), dirbase=RTKdir,
                                   void_classes=ut.void_classes['RTK'], n_classes=13)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    inp_image, inp_label = next(iter(val_loader))

    target_layers = [model.classifier.aspp.project, model.classifier.project]
    gc_model = GradCAMModel(model, target_layers)
    image0 = dp.get_image_from_input_tensor(inp_image, 0)
    image0_cam = CAMImage(image0)
    image0_cam.generate_cams(gc_model)
