from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.lib import shift_by
from functools import reduce
from operator import or_
import math
import jax
import dill
from dotenv import load_dotenv
import os

load_dotenv()
MAX_DIGITS = 4
MAX_LENGTH = 70
BOS = 'bos'
PAD = 'pad'

ZERO_SOP = rasp.Map(lambda _: 0, rasp.indices)

def generate_sum_operation():
    """
    compiles a model capable of doing sums up to MAX_DIGITS digits.
    """
    sop = rasp.tokens
    numbers = rasp.SequenceMap(
        lambda x, i: int(x) if x.isdigit() else 0, sop, rasp.indices
    ).named('numbers')
    plus_selector = rasp.Select(sop, rasp.Map(lambda x: '+', sop), rasp.Comparison.EQ).named('plus_selector')
    plus_pos = rasp.Aggregate(plus_selector, rasp.indices).named('plus_pos')
    equals_selector = rasp.Select(sop, rasp.Map(lambda x: '=', sop), rasp.Comparison.EQ).named('equals_selector')
    equals_pos = rasp.Aggregate(equals_selector, rasp.indices).named('equals_pos')
    indices = rasp.indices

    digit_sums = sum_by_digits(plus_pos, equals_pos, numbers)
    final_digits = percolate_carries(digit_sums)
    num_digits = count_total_digits(final_digits)
    final_result = assign_location_to_each_digit(final_digits, num_digits, equals_pos)

    return final_result

def compile_operation(expression, max_length=MAX_LENGTH):
    model = compiling.compile_rasp_to_model(
        expression,
        vocab={str(i) for i in range(0, 10)}.union({'+', '=', ' '}),
        max_seq_len=max_length,
        compiler_bos=BOS,
        compiler_pad=PAD,
        causal=True,
    )
    return model

def sum_by_digits(plus_pos, equals_pos, numbers):
    """
    we do a digit by digit sum. for example:
    [2, 5, 8]
    +
    [7, 7, 7]
    =
    [2, 12, 15]
    """
    digit_sums = []
    for d in range(MAX_DIGITS):
        def safe_left_idx_func(idx, d=d):
            if idx - d - 1 >= 0:
                return idx - d - 1
            return idx

        safe_left_idx_d = rasp.Map(safe_left_idx_func, plus_pos).named(f"safe_left_{d}_digit")

        def safe_right_idx_func(equals, plus, d=d):
            pos = equals - 1 - d
            if pos >= plus:
                return pos
            return plus


        safe_right_idx_d = rasp.SequenceMap(
            safe_right_idx_func,
            equals_pos, plus_pos,
        ).named(f"safe_right_{d}_digit")

        left_digit_d = rasp.Aggregate(
            rasp.Select(rasp.indices, safe_left_idx_d, rasp.Comparison.EQ),
            numbers
        ).named(f"left_{d}_digit")

        right_digit_d = rasp.Aggregate(
            rasp.Select(rasp.indices, safe_right_idx_d, rasp.Comparison.EQ),
            numbers
        ).named(f"right_{d}_digit")

        digit_sum_d = (left_digit_d + right_digit_d)
        digit_sums.append(digit_sum_d)

    return digit_sums

def percolate_carries(digit_sums):
    """
    we "percolate" the carries we obtained. for example:
    [2, 12, 15] -> [3, 3, 5]
    """
    carry_current = ZERO_SOP
    final_digits = []
    for d in range(MAX_DIGITS):
        final_digit_d = rasp.SequenceMap(
            lambda l, c: (l + c) % 10,
            digit_sums[d],
            carry_current
        ).named(f"final_{d}_digit")

        next_carry = rasp.SequenceMap(
            lambda l, c: (l + c) // 10,
            digit_sums[d],
            carry_current
        ).named(f"next_carry_{d}_digit")


        final_digits.append(final_digit_d)
        carry_current = next_carry

    return final_digits


def count_total_digits(final_digits):
    """
    we count how many digits does our final result have (to know where to put them later)
    [0, 2, 0, 1] -> 3
    """
    num_digits = ZERO_SOP
    has_digit = ZERO_SOP
    for d in list(range(MAX_DIGITS))[::-1]:
        has_digit = rasp.SequenceMap(lambda x, y: 1 if x + y > 0 else 0, final_digits[d], has_digit).named(f"has_{d}_digit")
        num_digits += has_digit

    return num_digits

def assign_location_to_each_digit(final_digits, num_digits, equals_pos):
    """
    all the digits are computed in every token. so we filter to leave only the correct digit
    according to the digits number
    """
    final_result = ZERO_SOP
    for d in range(MAX_DIGITS):

        def target_location_func(x, y, d=d):
            location = x + y - d - 1
            if location >= MAX_LENGTH or location <= 0:
                return 0

            return location

        target_location = rasp.SequenceMap(target_location_func, equals_pos, num_digits).named(f"target_{d}_location")
        tmp_mask = rasp.SequenceMap(lambda x, y: int(x == y), target_location, rasp.indices).named(f"tmp_mask_{d}_digit")
        select_off_by_offset = rasp.Select(target_location, rasp.indices, rasp.Comparison.EQ).named(f"offset_{d}_digit")
        digit_in_correct_place = (final_digits[d] * tmp_mask).named(f"digit_{d}_correct_place")
        maxed_sum = rasp.SequenceMap(lambda x, y: 0 if x + y >= 10 else x + y, final_result, digit_in_correct_place)
        final_result = maxed_sum.named(f"tmp_result_{d}_digit")
    return final_result.named('final_result')

def save_model(model, location=os.getenv('STORAGE_DIR'), filename='sum-model.dill'):
    filepath = location + '/' + filename
    filepath = os.path.expanduser(filepath)
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        dill.dump(model, f)
        print(f"model saved at: {filepath}")


if __name__ == '__main__':
    expression = generate_sum_operation()

    expression(['8', '8', '+', '3', '4', '=', '1', '2'])
    model = compile_operation(expression)
    param_count = sum(x.size for x in jax.tree_leaves(model.params))
    model.apply([BOS, '2', '+', '2', '=', '1', '2']).decoded
    result = model.apply([BOS, '8', '8', '+', '3', '4', '=', '1', '2']).decoded
    print('the result of adding 88 + 34 is', result[-3:])
    save_model(model)
