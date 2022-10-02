import numpy as np
import torch
import torchvision.transforms.functional as F
import utils as ut


class OptimumInput:
    def __init__(self, gc_model):
        self.gc_model = gc_model

    def generate(self, input_=None, class_ix=True, neuron_pos=None, lr=50, reg_l2=1e-3, n_its=1000,
                 input_shape=(1, 3, 352, 288), p_jitter=.5, p_blur=1, p_setnull=1):

        if input_ is None:
            input_ = torch.zeros(input_shape)
        else:
            input_shape = input_.shape

        best_logits = -np.inf
        list_res = []

        for it in range(n_its):
            input_ = input_.clone()

            fl_jitter = False
            if ut.decide_randomly(p_jitter):
                fl_jitter = True
                mark_h = torch.randint(size=(1,), high=input_.shape[2])
                mark_w = torch.randint(size=(1,), high=input_.shape[3])
                input_ = torch.concat([input_[:, :, mark_h:], input_[:, :, :mark_h]], axis=2)
                input_ = torch.concat([input_[..., mark_w:], input_[..., :mark_w]], axis=3)

            if ut.decide_randomly(p_blur):
                input_ = F.gaussian_blur(input_, kernel_size=3)

            input_.requires_grad = True
            self.gc_model.model.zero_grad()

            out_logits = self.gc_model(input_.cuda())[0]
            ix_logits = out_logits[class_ix].sum()
            ix_logits.backward(retain_graph=True)

            grad_out_wrt_input = input_.grad.clone()
            if neuron_pos is not None:
                grad_out_wrt_neuron = self.gc_model.grads[0][neuron_pos].sum().clone().cpu()
                grad_neuron_wrt_input = grad_out_wrt_input / (grad_out_wrt_neuron + 1e-8)
            else:
                grad_neuron_wrt_input = grad_out_wrt_input
            grad_term = grad_neuron_wrt_input / grad_neuron_wrt_input.norm()

            input_ = input_.detach()
            input_ = input_ + lr * grad_term - reg_l2 * input_

            # Set points of small grads or values to zero to avoid noise from them
            if it > 50 and ut.decide_randomly(p_setnull):
                small_value = 5e-3
                mask_small_values = (input_ < small_value) & (input_ > -small_value)
                small_value = 2e-4
                mask_small_grads = (grad_term < small_value) & (grad_term > -small_value)
                input_[mask_small_values & mask_small_grads] = 0

            # Undo jitter
            if fl_jitter:
                mark_w_rev = input_.shape[3] - mark_w
                input_ = torch.concat([input_[..., mark_w_rev:], input_[..., :mark_w_rev]], axis=3)
                mark_h_rev = input_.shape[2] - mark_h
                input_ = torch.concat([input_[:, :, mark_h_rev:], input_[:, :, :mark_h_rev]], axis=2)

            input_ = ut.augs.normalize_invert(input_).clip(0, 1)

            if ix_logits > best_logits:
                best_logits = ix_logits
                input_best = input_.clone()

            if it % (n_its // 10) == 0:
                list_res.append(input_[0].clone())

            input_ = ut.augs.normalize(input_)

        return input_best, list_res

    def generate_perclass(self, n_classes=12):
        list_res = []
        for i in range(n_classes):
            list_res.append(self.generate(class_ix=i)[0])
        return list_res
