from data.int_sum import prepare_sum_dataset
from data.int_mult import prepare_mult_dataset
from data.dyck_3 import prepare_dyck_dataset

def get_task(task_name, ood=False, ood_new_token=False, ood2=False, ood_length=False):
    if task_name == 'int-sum':
        return prepare_sum_dataset(ood=ood)
    if task_name == 'int-mult':
        return prepare_mult_dataset()
    if task_name == 'dyck':
        return prepare_dyck_dataset(ood=ood, ood_new_token=ood_new_token, ood2=ood2, length_gen=ood_length)

