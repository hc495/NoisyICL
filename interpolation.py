import copy
import torch

def reset_parameter(model):
    return_model = copy.deepcopy(model)
    return_model.apply(return_model._init_weights)
    return return_model


def model_linear_interpolation(
        model1, 
        model2, 
        model1_rate, 
        noise='None', 
        noise_intensity=0
    ):
    # Basically, if you let the noise_intensity from 0 to 1, the final model should be input in :model1.
    return_model = copy.deepcopy(model1)
    ret_dict = return_model.state_dict()

    if noise == 'None':
        for name, value in return_model.named_parameters():
            ret_dict[name] = model1.state_dict()[name] * model1_rate + model2.state_dict()[name] * (1 - model1_rate) 
        return_model.load_state_dict(ret_dict)
    if noise == 'Gaussian':
        for name, value in return_model.named_parameters():
            ret_dict[name] = model1.state_dict()[name] * model1_rate + model2.state_dict()[name] * (1 - model1_rate) + torch.randn_like(model1.state_dict()[name]) * noise_intensity * (1 - model1_rate)
        return_model.load_state_dict(ret_dict)

    return return_model


def model_linear_reduction(
        model1, 
        model1_rate, 
    ):
    # For the ablation study.
    return_model = copy.deepcopy(model1)
    ret_dict = return_model.state_dict()

    for name, value in return_model.named_parameters():
        ret_dict[name] = model1.state_dict()[name] * model1_rate
    return_model.load_state_dict(ret_dict)

    return return_model