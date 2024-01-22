import interpolation
import copy
import evaluate
import message_log
from transformers import AutoTokenizer, AutoModelForCausalLM

def main_experiment(pre_trained_model_name, pre_trained_tokenizer_name, dataset_loader, one_minus_lambdas=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0], tries=10, demos=[32,16,8,4,2,1,0], repeat = 5):
    # Main experiment of this paper. 
    # Parameter: 
    #   pre_trained_model_name: pre-trained model name from the Huggingface
    #   pre_trained_tokenizer_name: pre-trained tokenizer name from the Huggingface
    #   dataset_loader: DatasetLoader object from the dataset_loader.py
    #   one_minus_lambdas: (1—\lambda) s which are expected to be experimented. NOTICE: This is (1-\lambda), if you want to try a scenario with \lambda == 0.1, you should input 0.9
    #   tries: The repeat times for each query
    #   demos: Number of demos which are expected to be experimented
    #   repeat: Repeating times for each whole experiment
    # Output: .csv table of:
    #   1. *acc.csv: Accuracies table, amount: 1
    #   2. *F1.csv: MF1 table, amount: 1
    #   3. *ECE1.csv: ECE-1 table, amount: 1
    #   4. *output_table.csv: Table of [true label, predicted label, confidence], amount: len(lambdas) * len(demos)

    model_on_cpu = AutoModelForCausalLM.from_pretrained(pre_trained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    table_head = copy.deepcopy(demos)
    table_head.insert(0, tries)
    acc_matrix = [table_head]
    for i in range(0, len(one_minus_lambdas)):
        table_head = copy.deepcopy(table_head)
        table_head[0] = 1 - one_minus_lambdas[i]
        acc_matrix.append(table_head)
    F1_matrix = copy.deepcopy(acc_matrix)
    ECE1_matrix = copy.deepcopy(acc_matrix)
    
    model_zero = interpolation.reset_parameter(model_on_cpu)
    
    output_dirname = pre_trained_model_name + ', ' + dataset_loader.dataset_name
    output_dirname = message_log.new_folder(output_dirname)
    message_log.output_path += output_dirname + '/'
    
    for r in range(0, repeat):
        for i in range(0, len(one_minus_lambdas)):
            model = interpolation.model_linear_interpolation(model_on_cpu, model_zero, one_minus_lambdas[i]).cuda()
            for param in model.parameters():
                param.requires_grad = False

            for j in range(0, len(demos)):
                print("lambda: " + str(1 - one_minus_lambdas[i]) + "  demos: " + str(demos[j]))
                ACC, MF1, ECE1, output_table = evaluate.ICLAcc_evaluate(model, tokenizer, dataset_loader, demos[j], tries)
                acc_matrix[i+1][j+1] = ACC
                F1_matrix[i+1][j+1] = MF1
                ECE1_matrix[i+1][j+1] = ECE1
                message_log.output_single_csv(output_table, extra_name = 'lambda: ' + str(1 - one_minus_lambdas[i]) + ', demos: ' + str(demos[j]))
        del model
        
    message_log.output_single_csv(acc_matrix, extra_name = 'acc')
    message_log.output_single_csv(F1_matrix, extra_name = 'F1')
    message_log.output_single_csv(ECE1_matrix, extra_name = 'ECE1')