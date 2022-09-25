import numpy as np
import torch
import datapipe as dp
import torchvision.transforms.functional as F


class OptimumInput:
    def __init__(self, gc_model):
        self.gc_model = gc_model

    def generate(self, class_ix=True, neuron_pos=None, neuron_layer_ix=0, lr=50, reg_l2=1e-9, n_its=1000,
                 input_shape=(1, 3, 352, 288)):
        input_ = torch.zeros(input_shape)
        best_logits = -np.inf
        list_res = []

        for it in range(n_its):
            input_ = input_.clone()
            input_.requires_grad = True
            self.gc_model.model.zero_grad()

            out_logits = self.gc_model(input_.cuda())[0]
            ix_logits = out_logits[class_ix].sum()
            ix_logits.backward(retain_graph=True)

            if ix_logits > best_logits:
                best_logits = ix_logits
                input_best = input_.detach().clone()

            grad_out_wrt_input = input_.grad.clone()
            if neuron_pos is not None:
                grad_out_wrt_neuron = self.gc_model.grads[0][neuron_pos].clone().cpu()
                grad_neuron_wrt_input = grad_out_wrt_input / (grad_out_wrt_neuron + 1e-8)
            else:
                grad_neuron_wrt_input = grad_out_wrt_input

            grad_term = grad_neuron_wrt_input / grad_neuron_wrt_input.norm()
            grad_term = (grad_term - reg_l2 * input_)
            input_ = (input_ + lr * grad_term).detach()

            input_ = input_.detach()
            input_ = dp.augs.normalize_invert(input_)
            input_ = dp.augs.normalize(input_.clip(0, 1))

            input_ = F.gaussian_blur(input_, kernel_size=3)
            if it > (n_its / 10):
                small_value = 5e-3
                mask_small_values = (input_ < small_value) & (input_ > -small_value)
                small_value = 2e-4
                mask_small_grads = (grad_term < small_value) & (grad_term > -small_value)
                input_[mask_small_values & mask_small_grads] = 0

            if it % (n_its // 10) == 0:
                input_im = dp.augs.normalize_invert(input_)[0]
                list_res.append(input_im)

            return input_best, list_res

    def generate_perclass(self, n_classes=12):
        list_res = []
        for i in range(n_classes):
            list_res.append(self.generate(class_ix=i)[0])
        return list_res
