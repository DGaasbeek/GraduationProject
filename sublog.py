import os
import pandas as pd
import pm4py

path_in = "C:/Users/dgaas/PycharmProjects/GradProject Multi-task Transformer/datasets/Synthetic/Synthetic.xes"

log = pm4py.read_xes(path_in)

print(log.columns)

# subset = 'W'
# tasks = log['concept:name'].unique().tolist()
# subtask = subset + '_'
#
# tasks = [i for i in tasks if subtask in i]
# log = log[log['concept:name'].isin(tasks)]

# log['CompleteTime'] = log['time:timestamp']
log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])

log['StartTime'] = log.groupby('case:concept:name')['time:timestamp'].transform('min')
log['CompleteTime'] = log.groupby('case:concept:name')['time:timestamp'].transform('max')

case_summary = log.groupby('case:concept:name').agg(StartTime=('StartTime', 'min'),
                                       CompleteTime=('CompleteTime', 'max')).reset_index()

case_summary['Duration'] = (case_summary['CompleteTime'] - case_summary['StartTime']).dt.total_seconds() / (24 * 60 * 60)

print(case_summary['Duration'].mean())
print(case_summary['Duration'].max())