import os
import torch
import argparse
import json
from tqdm import tqdm
from datetime import date

from utils.load_kb import DataForSPARQL
from .data import DataLoader
from .model import SPARQLParser
from .sparql_engine import get_sparql_answer
from .preprocess import postprocess_sparql_tokens

from . import test_sparql_engine

from loguru import logger

import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query


def whether_equal(answer, pred):
    """
    check whether the two arguments are equal as attribute value
    """
    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = '{} {}'.format(str(v), ' '.join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split('-')
            y_split = y.split('-')
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred


def test(args, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    data = DataLoader(vocab_json, test_pt, 64, training=False)
    vocab = data.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))

    logger.info(f'Loading model: {model_name}')
    model = SPARQLParser(vocab, args.dim_word, args.dim_hidden, args.max_dec_len)
    model = model.to(device)
    if device == 'cpu':
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_name))
    
    count, correct = 0, 0

    logger.info(f"Start testing {model_name}")
    for batch in tqdm(data, total=len(data)):
        question, choices, sparql, answer = batch
        question = question.to(device)
        pred_sparql = model(question)

        pred_sparql = pred_sparql.cpu().numpy().tolist()
        answer = answer.cpu().numpy().tolist()
        for a, s in zip(answer, pred_sparql):
            given_answer = vocab['answer_idx_to_token'][a]
            s = [vocab['sparql_idx_to_token'][i] for i in s]
            end_idx = len(s)
            if '<END>' in s:
                end_idx = s.index('<END>')
            s = ' '.join(s[1:end_idx])
            s = postprocess_sparql_tokens(s)
            try:
                pred_answer = get_sparql_answer(s, kb)
            except Exception as e:
                # print('Error in testing when executing SPARQL query: \n{}'.format(s))
                pred_answer = None
            is_match = whether_equal(given_answer, pred_answer)
            count += 1
            if is_match:
                correct += 1
            answer = str(pred_answer)
    
    acc = correct / count
    logger.info(f'Test Accuracy of model {model_name}: {acc:.4f}\n')
    return acc


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path of all model checkpoints')

    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--max_dec_len', default=100, type=int)

    parser.add_argument('--mode', required=True, choices=["single", "multiple"], type=str)

    parser.add_argument('--results_dir', required=True, help='path to save test results')

    args = parser.parse_args()

    logger.add(os.path.join(args.results_dir, 'log.txt'), format="{time} {level:8} {message}")

    # args display
    for k, v in vars(args).items():
        logger.info(k + ': ' + str(v))
    
    try:
        test_sparql_engine.execute(disable_output=True)
    except Exception as e:
        logger.error(f'Error in test.py:main() when executing test_sparql_engine.execute(disable_output=True): \n{e}')
        logger.error('Please check whether the virtuoso server is running and the connection is correct.')
        exit(1)

    # Get model files
    pt_files = []
    if args.mode == "single":
        if not os.path.isfile(args.save_dir) or not args.save_dir.endswith('.pt'):
            logger.error('Error: In "single" mode, save_dir must be a path to a single .pt file.')
            exit(1)
        pt_files = [args.save_dir]
    elif args.mode == "multiple":
        if not os.path.isdir(args.save_dir):
            logger.error('Error: In "multiple" mode, save_dir must be a directory containing .pt files.')
            exit(1)
        pt_files = [os.path.join(args.save_dir, file) for file in os.listdir(args.save_dir) if file.endswith('.pt')]
        if not pt_files:
            logger.error('Error: No .pt files found in the specified save_dir directory.')
            exit(1)
    else:
        logger.error('Error: Invalid mode specified. Please choose "single" or "multiple".')
        exit(1)

    logger.info(f"Ready to load {len(pt_files)} model(s): {pt_files}")

    results = {}
    for model_name in pt_files:
        acc = test(args, model_name)
        results[model_name] = acc
    
    with open(os.path.join(args.results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

"""
Run in command line:

- 'single' mode:
    python -m SPARQL.test --input_dir processed_data --save_dir checkpoints/model_epoch0.pt --mode single --results_dir test_results

- 'multiple' mode:
    python -m SPARQL.test --input_dir processed_data --save_dir checkpoints --mode multiple --results_dir test_results
"""
if __name__ == '__main__':
    main()
