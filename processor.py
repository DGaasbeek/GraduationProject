import os
import sys
import pickle
import pandas as pd
import numpy as np
import itertools
import time
import argparse

from sklearn import preprocessing
from pandasql import sqldf
from datetime import datetime
from args import *
from tqdm import tqdm
import pm4py

class Processor():
    def __init__(self, MY_WORKSPACE_DIR, MILESTONE_DIR, args, enrich=False):
        self.enrich = enrich
        self.args = args
        self.dataset = args['log_name']
        self.MY_WORKSPACE_DIR = MY_WORKSPACE_DIR
        self.MILESTONE_DIR = MILESTONE_DIR
        self.milestone = args['milestone']
        self.max_size = args['max_size']
        self.min_size = args['min_size']

        self.path_to_sql = args['pref_sql']
        self.path_to_log = args['event_log']
        self.path_prefix_csv = args['prefix_file']
        self.path_prefixes_train = args['prefixes_train']
        self.path_prefixes_test = args['prefixes_test']

        fd = open(self.path_to_sql, 'r')
        self.queries = fd.read().split(';')
        fd.close()

    def process(self, cont_features, test_percentage=0.2): #todo allow for checking different test_percentages
        processed = True if os.path.isfile(self.args['processed_training_vec']) else False
        if processed:
            print('Dataset was processed earlier, loading saved files...')
            vec_train, vec_test, weights, indexes, pre_index = self.loader(self.args)
        else:
            print('Starting the preprocessing...')
            vec_train, vec_test, weights, indexes, pre_index = self.prefix_to_vec(cont_features=cont_features, test_percentage=test_percentage)

        return vec_train, vec_test, weights, indexes, pre_index

    def prefix_extraction(self, test_percentage=0.2):
        print('Loading logs and transforming traces...')
        self.log_to_traces()
        self.split_train_test(test_percentage)

        # self.df_test.role = self.df_test.role.astype(str) #otherwise, it saves 0 instead of 000

        print(self.args['test_traces'])
        self.df_test.to_csv(self.args['test_traces'])

        print('Running SQLite query to transform traces to prefixes...')
        self.prefixes_train = self.traces_to_prefix(self.df_train)
        self.prefixes_test = self.traces_to_prefix(self.df_test)
        self.prefixes_all = pd.concat([self.prefixes_train, self.prefixes_test])

        print('Saving the prefix files...')
        self.prefixes_train.to_csv(self.path_prefixes_train)
        self.prefixes_test.to_csv(self.path_prefixes_test)
        self.prefixes_all.to_csv(self.path_prefix_csv)

        return self.prefixes_train, self.prefixes_test

    def prefix_to_vec(self, cont_features, test_percentage=0.2):
        vec_train = []
        vec_test = []
        exists_pref = os.path.isfile(self.args['prefix_file'])

        print('Loading dataset...')
        if exists_pref:
            all_df = pd.read_csv(self.args['prefix_file'])
            all_df = all_df.drop('Unnamed: 0', axis=1).reset_index(drop=True)
            train_df = pd.read_csv(self.path_prefixes_train)
            train_df = train_df.drop('Unnamed: 0', axis=1).reset_index(drop=True)
            test_df = pd.read_csv(self.path_prefixes_test)
            test_df = test_df.drop('Unnamed: 0', axis=1).reset_index(drop=True)
        else:
            train_df, test_df = self.prefix_extraction(test_percentage)
            all_df = pd.read_csv(self.args['prefix_file'])
            all_df = all_df.drop('Unnamed: 0', axis=1).reset_index(drop=True)

        if self.milestone != 'All':
            # all_df = all_df[all_df['milestone'] == self.milestone]
            # train_df = train_df[train_df['milestone'] == self.milestone]
            test_df = test_df[test_df['milestone'] == self.milestone]
            # test_df.to_csv(self.args['test_traces'])

            #todo: save this one separately
        else:
            all_df = all_df[(all_df['prefix_length'] > self.min_size) & (all_df['prefix_length'] <= self.max_size)]
            train_df = train_df[
                (train_df['prefix_length'] > self.min_size) & (train_df['prefix_length'] <= self.max_size)]
            test_df = test_df[(test_df['prefix_length'] > self.min_size) & (test_df['prefix_length'] <= self.max_size)]

        all_df.loc[:, 'task'] = all_df['task'].astype(str)
        train_df.loc[:, 'task'] = train_df['task'].astype(str)
        test_df.loc[:, 'task'] = test_df['task'].astype(str)

        #Index encoding for categorical variables
        print('Index encoding...')
        ac_index = self.create_index(all_df, ['Task'.lower(), 'next_activity'])
        ac_index['start'] = 0
        index_ac = {v: k for k, v in ac_index.items()}
        train_df['ac_index'] = train_df['Task'.lower()].map(ac_index)
        test_df['ac_index'] = test_df['Task'.lower()].map(ac_index)

        if 'Role'.lower() in all_df.columns:
            print('Role in columns') #Sanity check
            rl_index = self.create_index(all_df, ['role'])
            rl_index['start'] = 0
            rl_index['end'] = len(rl_index)
            rl_index['unk'] = len(rl_index)
            index_rl = {v: k for k, v in rl_index.items()}
            train_df['rl_index'] = train_df['role'].map(rl_index)
            test_df['rl_index'] = test_df['role'].map(rl_index)
        else:
            rl_index = None
            index_rl = None

        ne_index = ac_index.copy()
        index_ne = {v: k for k, v in ne_index.items()}
        train_df['ne_index'] = train_df['next_activity'].map(ne_index)
        test_df['ne_index'] = test_df['next_activity'].map(ne_index)

        print('Standardizing and reformatting...')
        log_df_train, time_scaler1, y_scaler1 = self.standardization(train_df, cont_features)
        log_df_test, time_scaler2, y_scaler2 = self.standardization(test_df, cont_features)

        log_train = self.reformat_events(log_df_train, ac_index, rl_index, ne_index)
        log_test = self.reformat_events(log_df_test, ac_index, rl_index, ne_index)

        trc_len_train, cases_train = self.lengths(log_train)
        trc_len_test, cases_test = self.lengths(log_test)
        trc_len = max([trc_len_train, trc_len_test])

        print(f'Prefixes train: {cases_train}, Prefixes test: {cases_test}')

        vec_train = self.vectorization(log_train, ac_index, rl_index, ne_index=ne_index, trc_len=trc_len,
                                       cases=cases_train)
        vec_test = self.vectorization(log_test, ac_index, rl_index, ne_index=ne_index, trc_len=trc_len,
                                      cases=cases_test)

        sorted_index_ac = sorted(index_ac.keys())
        num_classes_ac = len(ac_index)
        ac_weights = np.eye(num_classes_ac)[sorted_index_ac]

        if rl_index != None:
            sorted_index_rl = sorted(index_rl.keys())
            num_classes_rl = len(rl_index)
            rl_weights = np.eye(num_classes_rl)[sorted_index_rl]
        else:
            rl_weights = None
            rl_index = None
            index_rl = None

        weights = {'ac_weights': ac_weights, 'rl_weights': rl_weights, 'next_activity': len(ne_index),
                   'y_scaler1': y_scaler1, 'y_scaler2': y_scaler2,
                   'time_scaler1': time_scaler1, 'time_scaler2': time_scaler2}
        indexes = {'index_ac': index_ac, 'index_rl': index_rl, 'index_ne': index_ne}
        pre_index = {'ac_index': ac_index, 'rl_index': rl_index, 'ne_index': ne_index}

        self.saver(vec_train, vec_test, weights, indexes, pre_index, self.args)

        return vec_train, vec_test, weights, indexes, pre_index #, y_scaler1, y_scaler2, time_scaler1, time_scaler2

    def import_rename(self):
        '''
        Function to read in the raw xes/csv file
        :return:
        '''
        print(self.path_to_log)

        if self.args['log_name'] == 'helpdesk':
            col_names = {'CaseID': 'caseid', 'ActivityID': 'task',
                         'CompleteTimestamp': 'end_timestamp'}
        else:
            col_names = {'case:concept:name': 'caseid', 'concept:name': 'task',
                         'time:timestamp': 'end_timestamp', 'org:resource': 'user'}

        if self.path_to_log.split('.')[1] == 'csv':
            self.log = pd.read_csv(self.path_to_log).rename(columns=col_names)
        elif self.path_to_log.split('.')[1] == 'gz':
            self.log = pm4py.read_xes(self.path_to_log)#.rename(columns=col_names)
            self.log = self.log.rename(columns=col_names)
        elif self.path_to_log.split('.')[1] == 'xes':
            self.log = pm4py.read_xes(self.path_to_log) #.rename(columns=col_names)
            self.log = self.log.rename(columns=col_names)

        # print(self.log.head(20))
        # print(self.log['lifecycle:transition'].unique())
        lifecycle = 'complete' if self.args['log_name'] == 'bpic17' else 'COMPLETE'
        if 'lifecycle:transition' in self.log.columns:
            self.log = self.log[self.log['lifecycle:transition'] == lifecycle]

        return self.log

    def log_to_traces(self):
        original_df = self.import_rename()

        q_raw, q_cases, q_traces, _,_,_ = self.queries

        raw = sqldf(q_raw, locals())
        if 'role' in raw.columns:
            raw.role = raw.role.fillna('0')
        raw['next_time'] = raw.groupby('caseid')['time_copy'].shift(-1)
        raw = raw.drop('time_copy', axis=1)

        if self.args['log_name'] == 'bpic12' or self.args['log_name'] == 'bpic17': #todo: make sure poac can be changed
            raw['seconds'] = (pd.to_datetime(raw['end_timestamp']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
            raw = raw.groupby('caseid').apply(self.assign_poac)

        cases = sqldf(q_cases, locals())
        self.traces = sqldf(q_traces, locals())

        return self.traces

    def split_train_test(self, test_percentage = 0.2, df = None):
        df = self.traces
        df['start_time'] = pd.to_datetime(df['start_time'], format = "%Y-%m-%d %H:%M:%S.%f", errors='ignore')
        df = df.sort_values(by='start_time', ascending=True)

        num_cases = df.caseid.unique()
        num_test_cases = int(np.round(len(num_cases) * test_percentage))

        unique_cases = pd.DataFrame(df['caseid'].unique(), columns=['caseid'])
        test_cases = unique_cases[-num_test_cases:]
        train_cases = unique_cases[:-num_test_cases]

        df_train = df.merge(train_cases, on='caseid')
        self.df_train = df_train.sort_values(['caseid', 'task_index'], ascending=(True, True))
        self.df_train = self.df_train.reset_index(drop=True)

        df_test = df.merge(test_cases, on='caseid')
        self.df_test = df_test.sort_values(['caseid', 'task_index'], ascending=(True, True))
        self.df_test = self.df_test.reset_index(drop=True)

        return self.df_train, self.df_test

    def traces_to_prefix(self, traces):
        '''
        Transformation: event log to prefix log including target variables
        :param raw:
        :param sql_path:
        :return: prefix log including target variables (dataframe)
        '''
        _, _, _, q_milestones, q_prefix, _ = self.queries

        milestones = self.get_milestones(traces)
        prefix_df = self.add_prefix_length(sqldf(q_prefix, locals()))
        prefix_df['next_time'].fillna(0, inplace=True)

        return prefix_df

    def get_milestones(self, traces):

        milestones = traces[['caseid', 'task_index', 'task']]

        milestones.loc[:, 'task'] = milestones['task'].astype(str)
        milestones.loc[:, 'next_activity'] = milestones.groupby('caseid')['task'].shift(-1).fillna('EOS')
        milestones.loc[:, 'milestone_id'] = milestones['task_index'].values
        milestones.rename(columns={'task': 'milestone', 'task_index': 'milestone_index'}, inplace=True)

        return milestones

    def add_prefix_length(self, prefix_df):
        df_temp = prefix_df.sort_values('task_index', ascending=False).drop_duplicates(['prefix_id'])

        prefix_lengths = df_temp[['prefix_id', 'task_index']]
        prefix_lengths.loc[:, 'prefix_length'] = prefix_lengths['task_index'].values
        prefix_lengths = prefix_lengths.drop('task_index', axis=1)

        final_df = prefix_df.join(prefix_lengths.set_index('prefix_id'), on='prefix_id')

        return final_df

    def assign_poac(self, group, poac='O_Returned'):
        # Check if the group contains the event O_SENT
        if poac in group['task'].values:
            # Get the index of the last occurrence of O_SENT in the group
            last_sent_index = group.loc[group['task'] == poac, 'seconds'].idxmax()
            # Assign 1 to rows before last occurrence of O_SENT and 0 to rows after
            group.loc[group.index <= last_sent_index, 'poac'] = 1
            group.loc[group.index > last_sent_index, 'poac'] = 0

            for i, row in group.iterrows():
                group.loc[i, 'poac_time'] = (group.loc[last_sent_index, 'seconds'] - row['seconds']) / 86400
                if i >= last_sent_index:
                    group.loc[i, 'poac_time'] = 0
        else:
            # Assign 0 to all rows if the group does not contain O_SENT
            group['poac'] = 0
            group['poac_time'] = 0

        return group

    def create_index(self, log, columns):

        concat = pd.concat([log[col] for col in columns]).drop_duplicates().reset_index(drop=True)
        alias = {value: i+1 for i, value in enumerate(concat)}

        return alias

    def standardization(self, log, features):
        time_scaler = preprocessing.StandardScaler()
        y_scaler = preprocessing.StandardScaler()

        for feature in features:
            if feature == 'next_time':
                log[feature] = y_scaler.fit_transform(log[feature].values.reshape(-1, 1)).astype(np.float64)
            else:
                log[feature] = time_scaler.fit_transform(log[feature].values.reshape(-1, 1)).astype(np.float64)

        return log, time_scaler, y_scaler

    def reformat_events(self, log_df, ac_index, rl_index=None, ne_index=None):
        """Creates series of activities, roles and relative times per trace.
            Author: Bemali Wickramanayake
        Args:
            log_df: dataframe.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        log_df = log_df.to_dict('records')
        poac = None
        temp_data = list()
        log_df = sorted(log_df, key=lambda x: (x['prefix_id'], x['task_index']))
        for key, group in itertools.groupby(log_df, key=lambda x: x['prefix_id']):
            trace = list(group)
            # dynamic features
            ac_order = [x['ac_index'] for x in trace]
            if rl_index != None:
                rl_order = [x['rl_index'] for x in trace]
            else: rl_order = None
            tbtw = [x['timelapsed'] for x in trace]

            # Reversing the dynamic feature order : Based on "An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism"
            ac_order.reverse()
            if rl_index != None: rl_order.reverse()
            tbtw.reverse()

            # outcome
            next_activity = max(x['ne_index'] for x in trace)
            next_time = max(x['next_time'] for x in trace)
            if self.args['log_name'] == 'bpic12' or self.args['log_name'] == 'bpic17':
                poac = max([x['poac'] for x in trace])
                poac_time = min(x['poac_time'] for x in trace)
                #Time to poac is measured from the milestone (last) event of the prefix
                #So, this event would also have the shortest time to the poac, hence min()


            temp_dict = dict(caseid=key,
                                 ac_order=ac_order,
                                 rl_order=rl_order,
                                 tbtw=tbtw,
                                 next_activity=next_activity,
                                 next_time=next_time,
                                poac=None,
                                poac_time=None
                              )

            if rl_index is not None: temp_dict['rl_order'] = rl_order
            if poac is not None:
                temp_dict['poac'] = poac
                temp_dict['poac_time'] = poac_time


            temp_data.append(temp_dict)

        return temp_data

    def lengths(self, log):
        """This function returns the maximum trace length (trc_len),
        and the number of cases for train and test sets (cases).
        The maximum out of trc_len for train and test sets will be
        used to define the trace length of the dataset that is fed to lstm.
        Author: Bemali Wickramanayake
        Args:
            log ([type]): [description]

        Returns:
            trc_len: maximum trace length
            cases: number of cases for train and test sets
        """
        trc_len = 1
        cases = 1

        for i, _ in enumerate(log):

            if trc_len < len(log[i]['ac_order']):

                trc_len = len(log[i]['ac_order'])
                cases += 1
            else:
                cases += 1

        return trc_len, cases

    def vectorization(self, log, ac_index, rl_index=None, ne_index=None, trc_len=0, cases=0):
        """Example function with types documented in the docstring.
            Author: Bemali Wickramanayake
        Args:
            #log: event log data in a dictionary.
            #ac_index (dict): index of activities.
            #rl_index (dict): index of roles (departments).
            #di_index (dict) : index of diagnosis codes.

        Returns:
            vec: Dictionary that contains all the LSTM inputs. """

        vec = {'prefixes': dict(),
               'next_activity': [],
               'next_time': [],
               'poac': [],
               'poac_time': []}
        len_ac = trc_len

        rl_exists = True if rl_index is not None else False

        vec['prefixes']['x_ac_inp'] = np.zeros((cases, len_ac))
        if rl_exists: vec['prefixes']['x_rl_inp'] = np.zeros((cases, len_ac))
        vec['prefixes']['xt_inp'] = np.zeros((cases, len_ac))
        vec['next_activity'] = np.zeros(cases)
        vec['next_time'] = np.zeros(cases)
        if self.args['log_name'] == 'bpic12' or self.args['log_name'] == 'bpic17':
            vec['poac'] = np.zeros(cases)
            vec['poac_time'] = np.zeros(cases)

        for i, _ in tqdm(enumerate(log)):
            padding = np.zeros(len_ac - len(log[i]['ac_order']))
            #print(log[i])
            vec['prefixes']['x_ac_inp'][i] = np.append(np.array(log[i]['ac_order']), padding)
            if rl_exists: vec['prefixes']['x_rl_inp'][i] = np.append(np.array(log[i]['rl_order']), padding)
            vec['prefixes']['xt_inp'][i] = np.append(np.array(log[i]['tbtw']), padding)
            vec['next_activity'][i] = int(log[i]['next_activity'])
            vec['next_time'][i] = log[i]['next_time']
            if self.args['log_name'] == 'bpic12':
                vec['poac'][i] = log[i]['poac']
                vec['poac_time'][i] = log[i]['poac_time']
        #One-hot encoding the y class
        vec['next_activity'] = np.eye(int(np.max(vec['next_activity']) + 1))[vec['next_activity'].astype(int)]
        if self.args['log_name'] == 'bpic12' or self.args['log_name'] == 'bpic17':
            vec['poac'] = np.eye(int(np.max(vec['poac']) + 1))[vec['poac'].astype(int)]


        return vec

    def saver(self, vec_train, vec_test, weights, indexes, pre_index, args):
        '''Helper function to save all the generated files

        :param vec_train:
        :param vec_test:
        :param weights:
        :param indexes:
        :param pre_index:
        :param args:
        :return: Nothing
        '''
        with open(args['processed_training_vec'], 'wb') as fp:
            pickle.dump(vec_train, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(args['processed_test_vec'], 'wb') as fp:
            pickle.dump(vec_test, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # converting the weights into a dictionary and saving
        with open(args['weights'], 'wb') as fp:
            pickle.dump(weights, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # converting the weights into a dictionary and saving
        with open(args['indexes'], 'wb') as fp:
            pickle.dump(indexes, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # converting the weights into a dictionary and saving
        with open(args['pre_index'], 'wb') as fp:
            pickle.dump(pre_index, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # saving the arguements (args)
        with open(args['args'], 'wb') as fp:
            pickle.dump(args, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def loader(self, args):
        '''
        Function to load all required processed files
        :param args:
        :return:
        '''
        with open(args['processed_training_vec'], 'rb') as fp:
            vec_train = pickle.load(fp)
        with open(args['processed_test_vec'], 'rb') as fp:
            vec_test = pickle.load(fp)
        with open(args['weights'], 'rb') as fp:
            weights = pickle.load(fp)
        with open(args['indexes'], 'rb') as fp:
            indexes = pickle.load(fp)
        with open(args['pre_index'], 'rb') as fp:
            pre_index = pickle.load(fp)

        return vec_train, vec_test, weights, indexes, pre_index