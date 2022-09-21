import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torch.autograd import Function
import datapipe as dp


def merge_cam_on_image(img, mask, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255
    cam = (1 - alpha) * heatmap + alpha * img
    return cam / np.max(cam)


class CAMImage:
    def __init__(self, image=None, input_=None):
        assert image is not None or input_ is not None, 'You should provide at least one argument.'
        self.image = image
        self.input_ = input_

        if self.image is None:
            self.image = dp.get_image_from_input_tensor(self.input_)

        if self.input_ is None:
            self.input_ = dp.get_input_tensor_from_image(self.image)
        self.input_.requires_grad =True

    def generate_cams(self, gc_model):
        device = next(gc_model.model.parameters()).device
        out_logits = gc_model(self.input_.to(device))[0]
        self.out_preds = out_logits.argmax(0)
        self.out_classes = torch.unique(self.out_preds).cpu().numpy().tolist()
        self.grads_input = {ix: None for ix in self.out_classes}
        self.heatmap = {ix: None for ix in self.out_classes}
        self.heatmap_merged = {ix: None for ix in self.out_classes}

        for ix_cl in self.out_classes:
            gc_model.model.zero_grad()
            # Backpropagate class logits
            ix_logits = out_logits[ix_cl, self.out_preds == ix_cl]
            ix_logits.sum().backward(retain_graph=True)
            self.grads_input[ix_cl] = self.input_.grad[0].clone()
            self.heatmap[ix_cl] = gc_model.compute_cam()[0].cpu().detach().numpy()
            self.heatmap_merged[ix_cl] = merge_cam_on_image(self.image, self.heatmap[ix_cl])

    def prepare_plot_line(self):
        n_classes = len(self.out_classes)
        fig, axs = plt.subplots(1, n_classes, figsize=((4 * n_classes) // 1, 5))
        [a.set_axis_off() for a in axs]
        return fig, axs

    def plot_heatmap_line(self):
        fig, axs = self.prepare_plot_line()
        for a, (k, v) in zip(axs, self.heatmap_merged.items()):
            a.set_title('class: {}'.format(k))
            a.imshow(v)

    def plot_input_saliency(self, fl_abs=True, fl_gentle=False, gentle_bound=.2):
        fig, axs = self.prepare_plot_line()
        for a, (k, v) in zip(axs, self.grads_input.items()):
            v = v.permute(1, 2, 0).detach().cpu().numpy()
            if fl_abs:
                v = np.abs(v).max(axis=2)
            else:
                v = v.clip(min=0).max(axis=2)
            if not fl_gentle:
                v = (v - v.min()) / (v.max() - v.min())
            else:
                v_min, v_max = np.quantile(v, [gentle_bound, 1 - gentle_bound])
                v = (v - v_min) / (v_max - v_min)
                v = v.clip(0, 1)
            a.set_title('class: {}'.format(k))
            a.imshow(v, cmap='gray')


class GradCAMModel:
    def __init__(self, model, target_layers, fl_guided=False, **kwargs):
        self.model = model
        if fl_guided:
            self.model.apply(lambda m: replace_layer(m, torch.nn.modules.activation.ReLU, GuidedReLU()));

        self.target_layers = target_layers
        self.activations = self.instantiate_list()
        self.grads = self.instantiate_list()
        self.cams = self.instantiate_list()

        for i, l in enumerate(target_layers):
            l.register_forward_hook(self.save_activation(i))
            l.register_full_backward_hook(self.save_grad(i))

        self.input_shape = None

    def instantiate_list(self):
        return [None] * len(self.target_layers)

    def save_activation(self, i):
        def save_layer_activation(module, input, output):
            self.activations[i] = output.clone()
        return save_layer_activation

    def save_grad(self, i):
        def save_layer_grad(module, grad_input, grad_output):
            self.grads[i] = grad_output[0].clone()
        return save_layer_grad

    def compute_cam(self):
        for i in range(len(self.activations)):
            self.cams[i] = torch.mean(self.grads[i], axis=[2, 3], keepdim=True) * self.activations[i]
            # ReLU operation
            self.cams[i] = self.cams[i].sum(axis=1).clamp(min=0)

        class_cam = [F.resize(x, self.input_shape) for x in self.cams]
        class_cam = torch.stack(class_cam).mean(0)
        # Max-min scale
        return (class_cam - class_cam.min()) / (class_cam.max() - class_cam.min())

    def __call__(self, inputs):
        self.input_shape = inputs.shape[2:]
        return self.model(inputs)


class GuidedReLUFunction(Function):
    @staticmethod
    def forward(ctx, input_):
        output = input_ * (input_ > 0)
        ctx.save_for_backward(input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        return grad_output * (grad_output > 0) * (input_ > 0)


class GuidedReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GuidedReLUFunction.apply(x)


def replace_layer(module, old_layer, new_layer):
    for name, m in module.named_children():
        if isinstance(m, old_layer):
            setattr(module, name, new_layer)


class GuidedModel(GradCAMModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.apply(lambda m: replace_layer(m, torch.nn.modules.activation.ReLU, GuidedReLU()));


def revert_guidedmodel(guided_model):
    return guided_model.apply(lambda m: replace_layer(m, GuidedReLU, torch.nn.modules.activation.ReLU()));
