from data.int_sum import prepare_sum_dataset
from data.int_mult import prepare_mult_dataset

def get_task(task_name):
    if task_name == 'int-sum':
        return prepare_sum_dataset()
    if task_name == 'int-mult':
        return prepare_mult_dataset()

