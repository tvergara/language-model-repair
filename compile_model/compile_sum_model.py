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
MAX_DIGITS = 5
MAX_LENGTH = 16
BOS = 'bos'
PAD = 'pad'


def generate_sum_operation():
    """
    compiles a model capable of doing sums up to MAX_DIGITS digits.
    """
    sop = rasp.tokens
    numbers = rasp.SequenceMap(lambda x, i: int(x) if x.isdigit() else 0, sop, rasp.indices)
    plus_selector = rasp.Select(sop, rasp.Map(lambda x: '+', sop), rasp.Comparison.EQ)
    plus_pos = rasp.Aggregate(plus_selector, rasp.indices)
    equals_selector = rasp.Select(sop, rasp.Map(lambda x: '=', sop), rasp.Comparison.EQ)
    equals_pos = rasp.Aggregate(equals_selector, rasp.indices)
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
    left_last = (plus_pos - 1)
    digit_sums = []
    for d in range(MAX_DIGITS):
        left_idx_d = (left_last - d)
        right_idx_d = ((equals_pos - 1) - d)

        safe_left_idx_d = rasp.SequenceMap(
            lambda idx, plus: idx if idx >= 0 else plus,
            left_idx_d, plus_pos
        )

        safe_right_idx_d = rasp.SequenceMap(
            lambda idx, plus: idx if idx >= plus else plus,
            right_idx_d, plus_pos
        )

        left_digit_d = rasp.Aggregate(
            rasp.Select(rasp.indices, safe_left_idx_d, rasp.Comparison.EQ),
            numbers
        )

        right_digit_d = rasp.Aggregate(
            rasp.Select(rasp.indices, safe_right_idx_d, rasp.Comparison.EQ),
            numbers
        )

        digit_sum_d = (left_digit_d + right_digit_d)
        digit_sums.append(digit_sum_d)

    return digit_sums

def percolate_carries(digit_sums):
    """
    we "percolate" the carries we obtained. for example:
    [2, 12, 15] -> [3, 3, 5]
    """
    carry_current = rasp.Map(lambda _: 0, rasp.indices)
    final_digits = []
    for d in range(MAX_DIGITS):
        sum_with_carry = (digit_sums[d] + carry_current)

        final_digit_d = rasp.SequenceMap(
            lambda val, i: val % 10,
            sum_with_carry, rasp.indices
        )

        next_carry = rasp.SequenceMap(
            lambda val, i: val // 10,
            sum_with_carry, rasp.indices
        )

        final_digits.append(final_digit_d)
        carry_current = next_carry

    return final_digits


def count_total_digits(final_digits):
    """
    we count how many digits does our final result have (to know where to put them later)
    [0, 2, 0, 1] -> 3
    """
    num_digits = rasp.Map(lambda x: 0, rasp.tokens)
    has_digit = rasp.Map(lambda x: 0, rasp.tokens)
    for d in list(range(MAX_DIGITS))[::-1]:
        has_digit = rasp.SequenceMap(lambda x, y: 1 if x + y > 0 else 0, final_digits[d], has_digit)
        num_digits += has_digit

    return num_digits

def assign_location_to_each_digit(final_digits, num_digits, equals_pos):
    """
    all the digits are computed in every token. so we filter to leave only the correct digit
    according to the digits number
    """
    final_result = rasp.Map(lambda x: 0, rasp.tokens)
    for d in range(MAX_DIGITS):
        target_location = equals_pos + num_digits - d - 1
        tmp_mask = rasp.SequenceMap(lambda x, y: int(x == y), target_location, rasp.indices)
        select_off_by_offset = rasp.Select(target_location, rasp.indices, rasp.Comparison.EQ)
        digit_in_correct_place = final_digits[d] * tmp_mask
        final_result += digit_in_correct_place
    return final_result

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
    model = compile_operation(expression)
    result = model.apply([BOS, '8', '8', '+', '3', '4', '=', '1', '2']).decoded
    print('the result of adding 88 + 34 is', result[-3:])
    save_model(model)

