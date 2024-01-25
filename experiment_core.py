import torch
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import MulticlassCalibrationError
import numpy as np
import prompting
from tqdm.notebook import tqdm as tqdm 
import numpy as np
from scipy.stats import entropy

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def ICLAcc_evaluate(
    model, 
    tokenizer, 
    dataset, 
    demos_amount, 
    tries=1, 
):
    torch.cuda.empty_cache()
    ECE1_function = MulticlassCalibrationError(num_classes=len(dataset.label_space), n_bins=10, norm='l1')
    total_count = 0
    correct_count = 0
    bar_format = '{percentage:3.0f}%|{n_fmt}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
    tqdm_ICL = tqdm(total=dataset.get_max(), bar_format=bar_format)
    
    true_labels = []
    predicted_labels = []
    confidences_amoung_label_spaces = []
    output_table = [['true label', 'predicted label', 'confidence']]
    MF1 = 0
    
    for i in range(0, dataset.get_max()):
        for j in range(0, tries):
            torch.cuda.empty_cache()
            prompt, true_label = prompting.default_prompting(dataset, demos_amount, i)
            tokenized_input = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            result_vector = model(tokenized_input)['logits'][0][-1]
            total_count += 1
            label_space_p = []
            for labels in dataset.label_space:
                label_space_p.append(result_vector[tokenizer(labels, return_tensors="np").input_ids[0][-1]].cpu().detach().item())
            predicted_label = np.argmax(label_space_p)
            true_label_index = dataset.label_space.index(true_label)
            if true_label_index == predicted_label:
                correct_count += 1
            label_space_p = softmax(label_space_p)

            true_labels.append(true_label_index)
            predicted_labels.append(predicted_label)
            confidence = max(label_space_p)
            confidences_amoung_label_spaces.append(label_space_p)
            output_table.append([true_label_index, predicted_label, confidence])
            
            MF1 = multiclass_f1_score(
                torch.tensor(predicted_labels), 
                torch.tensor(true_labels), 
                num_classes = len(dataset.label_space), 
                average = 'macro'
            ).item()
            ECE1 = ECE1_function(torch.Tensor(confidences_amoung_label_spaces), torch.Tensor(true_labels)).item()
            
            del tokenized_input
            del label_space_p
            del result_vector
            tqdm_ICL.set_postfix({
                'accuracy': '{0:1.4f}'.format(correct_count / total_count),
                'F1': '{0:1.4f}'.format(MF1),
                'ECE1': '{0:1.4f}'.format(ECE1)})
        tqdm_ICL.update(1)
    
    return correct_count / total_count, MF1, ECE1, output_table


def empty_query_entropy_evaluate(
    model, 
    tokenizer, 
    dataset, 
    demos_amount, 
    total_tries=1024, 
):
    torch.cuda.empty_cache()
    output_table = []
    for i in range(0, total_tries):
        torch.cuda.empty_cache()
        prompt, true_label = prompting.default_prompting(dataset, demos_amount - 1, -1)
        fake_prompt = prompt + true_label + "\nInput: " + dataset.get_empty_input()[0] + ", Label: "
        tokenized_input = tokenizer(fake_prompt, return_tensors="pt").input_ids.cuda()
        result_vector = model(tokenized_input)['logits'][0][-1]
        label_space_p = []
        for labels in dataset.label_space:
            label_space_p.append(result_vector[tokenizer(labels, return_tensors="np").input_ids[0][-1]].cpu().detach().item())
        label_space_p = softmax(label_space_p)
        res = entropy(label_space_p)
        output_table.append(res / np.log(len(dataset.label_space)))
        del(tokenized_input)
    return np.mean(output_table), np.std(output_table), output_table