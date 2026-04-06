import math
import pandas as pd
import numpy as np

# CLI
from argparse import ArgumentParser

from utils.models import get_model
from utils.strategy import resolve_strategy_kwargs
from utils.consts import RANDOM_STATE, strategies

BATCHSIZE = 1000

parser = ArgumentParser(
  prog='ŚMIGIEL GEN CLI',
  description='LLM-aided text generation helper'
)

parser.add_argument('--input_file_path', default='input.csv')
parser.add_argument('--job_dir')
parser.add_argument('--model', default='llama-sm')
parser.add_argument('--batch-size', type=int, default=BATCHSIZE)
parser.add_argument('--subset_start_index', default=None, type=int)
parser.add_argument('--subset_end_index', default=None, type=int)

args = parser.parse_args()

model = get_model(args.model)

# 1 establish the median
the_input_data = pd.read_csv(args.input_file_path)
print(f"input data length: {len(the_input_data)}")

the_input_data["tokens"] = the_input_data["text"].apply(lambda x: model.tokenize(x).shape[-1])
print(the_input_data.head())
median = int(np.median(the_input_data["tokens"]))
print(f"median: {median}")

def resolve_token_length(original_text_token_len, domain):
    # condition both max and min tokens basing on original text. 80-120%
    min_factor = 1. if domain == 'social' else .8
    max_factor = 2. if domain == 'social' else 1.2
    return {
        "min_new_tokens": math.ceil(original_text_token_len * min_factor),
        "max_new_tokens": math.ceil(original_text_token_len * max_factor)
    }


subset = the_input_data.iloc[args.subset_start_index:args.subset_end_index + 1].copy()

# resolve strategies
batch_strategies = pd.Series(strategies)
batch_strategies = batch_strategies.sample(n=len(subset), replace=True, random_state=RANDOM_STATE).tolist()

print(f"strategies: {batch_strategies}")

responses = [model.infer(prompt,
                            **resolve_strategy_kwargs(strategy),
                            **resolve_token_length(
                                model.tokenize(original_text).shape[-1],
                                domain,
                        )) for prompt, original_text, domain, strategy in zip(subset["prompt"], subset["text"], subset["domain"], batch_strategies)]
subset["strategy"] = batch_strategies
subset["gen"] = responses
subset["model"] = args.model
subset.to_csv(f'{args.job_dir}{args.model}_{args.subset_start_index}-{args.subset_end_index}.csv', index=True)