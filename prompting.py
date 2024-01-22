import random

def default_prompting(
        dataset, 
        demos_amount, 
        query_index = -1
    ):
    sample_list = []
    if query_index == -1:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount + 1)
    else:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        if query_index in sample_list:
            sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        sample_list.append(query_index)
    
    prompt = ''
    query_label = 0
    
    for i in range(0, demos_amount):
        text, label = dataset.get(sample_list[i])
        prompt += 'Input: '
        prompt += text
        prompt += ', Label: '
        prompt += label
        prompt += '\n'
    text, label = dataset.get(sample_list[-1])
    prompt += 'Input: '
    prompt += text
    prompt += ', Label: '
    query_label = label

    return prompt, query_label