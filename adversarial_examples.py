import datapipe as dp


def generate_adversarial_examples(model, input_, n_its=10, lr=1, n_classes=12, n_examples=5):
    list_adversarials = []

    for it in range(n_its):
        input_ = input_.clone()
        input_.requires_grad = True
        model.zero_grad()

        out_logits = model(input_.cuda())[0]
        for class_ix in range(n_classes):
            ix_logits = out_logits[class_ix][out_logits.argmax(0) == class_ix].sum()
            ix_logits.backward(retain_graph=True)

        grad_neuron_wrt_input = input_.grad.clone()
        grad_term = grad_neuron_wrt_input / grad_neuron_wrt_input.norm()
        input_ = (input_ - lr * grad_term).detach()
        input_ = dp.augs.normalize_invert(input_)
        input_ = dp.augs.normalize(input_.clip(0, 1))

        if it % (n_its // n_examples) == 0:
            list_adversarials.append(dp.augs.normalize_invert(input_)[0])

        return list_adversarials
