import os
import torch
import argparse
import json
from tqdm import tqdm

from utils.load_kb import DataForSPARQL
from .data import DataLoader
from .model import SPARQLParser
from .sparql_engine import get_sparql_answer
from .preprocess import postprocess_sparql_tokens

from . import test_sparql_engine

from loguru import logger

import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('load test data')
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    data = DataLoader(vocab_json, test_pt, 128, training=False)
    vocab = data.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))

    print('load model')
    model = SPARQLParser(vocab, args.dim_word, args.dim_hidden, args.max_dec_len)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt')))
    
    f = open(os.path.join(args.save_dir, 'predict.txt'), 'w')
    for batch in tqdm(data, total=len(data)):
        question, choices, sparql, answer = batch
        question = question.to(device)
        pred_sparql = model(question)

        pred_sparql = pred_sparql.cpu().numpy().tolist()
        for s in pred_sparql:
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
            answer = str(pred_answer)
            f.write(answer + '\n')
    f.close()



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')

    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--max_dec_len', default=100, type=int)
    args = parser.parse_args()

    logger.add(os.path.join(args.save_dir, 'log.txt'), format="{time} {level:8} {message}")

    if args.virtuoso_enabled.lower() == "true":
        try:
            test_sparql_engine.execute(disable_output=True)
        except Exception as e:
            logger.error(f'Error in train:main() when executing test_sparql_engine.execute(disable_output=True): \n{e}')
            logger.error('Please check whether the virtuoso server is running and the connection is correct.')
            exit(1)

    test(args)


if __name__ == '__main__':
    main()
