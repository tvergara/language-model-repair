from data.int_sum import prepare_sum_dataset

def get_task(task_name):
    if task_name == 'int-sum':
        return prepare_sum_dataset()
    pass

