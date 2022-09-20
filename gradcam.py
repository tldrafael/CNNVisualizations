import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import datapipe as dp


def merge_cam_on_image(img, mask, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255
    cam = (1 - alpha) * heatmap + alpha * img
    return cam / np.max(cam)


class CAMImage:
    def __init__(self, image=None, image_input=None):
        assert image is not None or image_input is not None, 'You should provide at least one argument.'
        self.image = image
        self.image_input = image_input

        if self.image is None:
            self.image = dp.get_image_from_input_tensor(self.image_input)

        if self.image_input is None:
            self.image_input = dp.get_input_tensor_from_image(self.image)
        self.image_input.requires_grad =True

    def generate_cams(self, gc_model):
        device = next(gc_model.model.parameters()).device
        out_logits = gc_model(self.image_input.to(device))[0]
        out_preds = out_logits.argmax(0)
        self.out_classes = torch.unique(out_preds).cpu().numpy().tolist()
        self.grads_input = {ix: None for ix in self.out_classes}
        self.heatmap = {ix: None for ix in self.out_classes}
        self.heatmap_merged = {ix: None for ix in self.out_classes}

        for ix_cl in self.out_classes:
            gc_model.model.zero_grad()
            # Backpropagate class logits
            ix_logits = out_logits[ix_cl, out_preds == ix_cl]
            ix_logits.sum().backward(retain_graph=True)
            self.grads_input[ix_cl] = self.image_input.grad[0].clone()
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

    def plot_input_saliency(self):
        fig, axs = self.prepare_plot_line()
        for a, (k, v) in zip(axs, self.grads_input.items()):
            v = v.permute(1, 2, 0).detach().cpu().numpy()
            # v = np.abs(v).max(axis=2)
            v = v.clip(min=0).max(axis=2)
            v = (v - v.min()) / (v.max() - v.min())
            a.set_title('class: {}'.format(k))
            a.imshow(v, cmap='gray')


class GradCAMModel:
    def __init__(self, model, target_layers):
        self.model = model
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
