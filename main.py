from data.get_task import get_task

TASK = 'int-sum'

train_dataset, test_dataset = get_task(TASK)

[x for x in train_dataset]

