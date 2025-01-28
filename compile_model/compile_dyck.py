from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.lib import shift_by, make_pair_balance, make_length, make_shuffle_dyck
from functools import reduce
from operator import or_
import math
import jax
import dill
from dotenv import load_dotenv
import os

load_dotenv()
MAX_LENGTH = 100
BOS = 'bos'
PAD = 'pad'

def shuffle_dyck(
    pairs=[('(', ')'), ('{', '}'), ('[', ']')]
):
    all_diffs = []
    all_negs = []
    for left, right in pairs:
        def compare_left(x, y, left=left):
            return x == left
        starts = rasp.Select(rasp.tokens, rasp.tokens, compare_left)
        start_counts = rasp.SelectorWidth(starts)

        def compare_right(x, y, right=right):
            return x == right
        ends = rasp.Select(rasp.tokens, rasp.tokens, compare_right)
        end_counts = rasp.SelectorWidth(ends)

        diffs = start_counts - end_counts
        all_diffs.append(diffs)

        negs_selector = rasp.Select(diffs, rasp.tokens, lambda x, y: x < 0)
        negs_counter = rasp.SelectorWidth(negs_selector)
        all_negs.append(negs_counter)

    current_negs = all_negs[0]
    for negs in all_negs[1:]:
        current_negs = rasp.SequenceMap(lambda x, y: 1 if (x != 0 or y != 0) else 0, current_negs, negs)

    current_diffs = all_diffs[0]
    for diffs in all_diffs[1:]:
        current_diffs = rasp.SequenceMap(lambda x, y: 1 if (x != 0 or y != 0) else 0, current_diffs, diffs)

    result = rasp.SequenceMap(lambda x, y: x == 0 and y == 0, current_negs, current_diffs)
    filtered_result = rasp.SequenceMap(lambda x, y: x if y == 'compute' else False, result, rasp.tokens)
    return filtered_result

def compile_operation(expression, max_length=MAX_LENGTH):
    model = compiling.compile_rasp_to_model(
        expression,
        vocab={'(', ')', ' ', 'compute'},
        max_seq_len=max_length,
        compiler_bos=BOS,
        compiler_pad=PAD,
        causal=True,
    )
    return model

def save_model(model, location=os.getenv('STORAGE_DIR'), filename='dyck-model.dill'):
    filepath = location + '/' + filename
    filepath = os.path.expanduser(filepath)
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        dill.dump(model, f)
        print(f"model saved at: {filepath}")

if __name__ == '__main__':
    expression = shuffle_dyck()
    model = compile_operation(expression)
    model.apply([BOS, '(', ')', 'compute'])
    param_count = sum(x.size for x in jax.tree_leaves(model.params))
    save_model(model)



