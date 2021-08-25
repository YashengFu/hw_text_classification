import torch


@torch.no_grad()
def eval_model(model,valid_data_loader):
    """evaluate the precision of the model

    Args:
        model : network
        valid_data_loader     : valid data loader
    """
    device = next(model.parameters()).device
    label_0_scores = []
    label_1_scores = []
    label_2_scores = []
    true_label = []
    for inputs,label in valid_data_loader:
        outputs = model(inputs.to(device))
        for i in range(len(label)):
            true_label.append(label[i].item())
            label_0_scores.append(outputs[i][0].item())
            label_1_scores.append(outputs[i][1].item())
            label_2_scores.append(outputs[i][2].item())
    return label_0_scores,label_1_scores,label_2_scores,true_label