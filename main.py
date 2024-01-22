import experiment_cell
import dataset_loader
import argparse

# Main experiment of this paper.

# Runtime parameter setting:
parser = argparse.ArgumentParser(
    description='NoisyICL Experiment Code for Papar: NoisyICL: A Little Noise in Model Parameters Calibrates In-context Learning'
)
parser.add_argument('-m', '--model', action = "store", type=str, default='gpt2', help='pre-trained model name from the Huggingface')
parser.add_argument('-d', '--dataset', action = "store", type=str, default='hate_speech18', help='dataset name from the Huggingface')
parser.add_argument('-v', '--validation', action = "store_true", help='validation or not (if not, test, and you must give a lambda to be tested)')
parser.add_argument('-l', '--noisy_intensity', action = "store", type=float, default=0, help='lambda (not 1-lambda)')
parser.add_argument('-e', '--experiment', action = "store", type=str, default='main', help='experiment name')
parser.add_argument('-t', '--tries', action = "store", type=int, default=2, help='tries')
parser.add_argument('-r', '--repeat', action = "store", type=int, default=5, help='repeat')

args = parser.parse_args()

# Model selection:
model_name = args.model
demos = []
if model_name == 'gptj' or model_name == 'GPT-J' or model_name == 'gpt-j':
    model_name = 'EleutherAI/gpt-j-6B'
    demos = [16,8,4,2,1,0]
elif model_name == 'gpt-2' or model_name == 'GPT2':
    model_name = 'gpt2'
    demos = [4,2,1,0]

# Dataset selection:
dataset = None
dataset_name = args.dataset
if dataset_name == 'hate_speech18' or dataset_name == 'HS':
    dataset = dataset_loader.hate_speech18()
elif dataset_name == 'poem_sentiment' or dataset_name == 'PS':
    dataset = dataset_loader.poem_sentiment()
elif dataset_name == 'SemEval 2014-Task 4 Restaurants' or dataset_name == 'SER':
    dataset = dataset_loader.SemEval2014_Restaurants()
elif dataset_name == 'SemEval 2014-Task 4 Laptops' or dataset_name == 'SEL':
    dataset = dataset_loader.SemEval2014_Laptops()
elif dataset_name == 'GLUE-RTE' or dataset_name == 'RTE':
    dataset = dataset_loader.glue_rte()
elif dataset_name == 'GLUE-MRPC' or dataset_name == 'MRPC':
    dataset = dataset_loader.glue_mrpc()
elif dataset_name == 'Ethos':
    dataset = dataset_loader.ethos()
elif dataset_name == 'financial_phrasebank' or dataset_name == 'FP':
    dataset = dataset_loader.financial_phrasebank()
elif dataset_name == 'GLUE-SST2' or dataset_name == 'sst2':
    dataset = dataset_loader.glue_sst2()
elif dataset_name == 'tweet_eval_emotion' or dataset_name == 'TEE':
    dataset = dataset_loader.tweet_eval_emotion()
elif dataset_name == 'tweet_eval_sentiment' or dataset_name == 'TES':
    dataset = dataset_loader.tweet_eval_sentiment()
elif dataset_name == 'tweet_eval_hate' or dataset_name == 'TEH':
    dataset = dataset_loader.tweet_eval_hate()

# Dataset dividing:
if args.validation:
    dataset.default_training_division()
else:
    dataset.default_testing_division()

# Experiment:
if args.experiment == 'main' and args.validation:
    experiment_cell.main_experiment(
        model_name, 
        model_name, 
        dataset, 
        one_minus_lambdas = [0.7, 0.8, 0.9, 1, 0.98, 0.96, 0.94, 0.92, 0.982, 0.984, 0.986, 0.988, 0.99, 0.992, 0.994, 0.996, 0.998], 
        tries = args.tries, 
        demos = demos, 
        repeat = args.repeat
    )
elif args.experiment == 'main' and not args.validation:
    experiment_cell.main_experiment(
        model_name, 
        model_name, 
        dataset, 
        one_minus_lambdas = [args.noisy_intensity], 
        tries = args.tries, 
        demos = demos, 
        repeat = args.repeat
    )