import os
import pickle

def _params_bpic12(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE,
                   N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET=None, ENRICH=False):

    parameters = dict()
    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")

    parameters['max_size'] = MAX_SIZE
    parameters['min_size'] = MIN_SIZE
    parameters['n_size'] = N_SIZE
    parameters['subset'] = SUBSET

    parameters['event_log'] = os.path.join(MY_WORKSPACE_DIR, 'BPI_Challenge_2012.xes.gz') #Change
    if SUBSET != None:
        parameters['event_log'] = os.path.join(os.path.join(os.getcwd(), 'BPIC12'), 'BPIC_2012_log.csv')

    parameters['prefix_file'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC_2012_Prefixes_all.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR, 'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR, 'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p')
    parameters['args'] = os.path.join(MILESTONE_DIR, 'args.p')
    parameters['milestone'] = MILESTONE
    parameters['prefix_length'] = 'fixed'

    parameters['pref_sql'] = os.path.join(MY_WORKSPACE_DIR, 'prefix_extraction.sql')
    parameters['test_traces'] = os.path.join(MY_WORKSPACE_DIR, 'bpic12_test.csv')
    parameters['prefixes_train'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC12_Prefixes_train.csv')
    parameters['prefixes_test'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC12_Prefixes_test.csv')
    parameters['suff_pred_res'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC12_suffpred_results.csv')

    parameters['log_name'] = 'bpic12'

    return parameters

def _params_bpic12W(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE,
                   N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET=None, ENRICH=False):

    parameters = dict()
    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")

    parameters['max_size'] = MAX_SIZE
    parameters['min_size'] = MIN_SIZE
    parameters['n_size'] = N_SIZE
    parameters['subset'] = SUBSET

    parameters['event_log'] = os.path.join(MY_WORKSPACE_DIR, 'BPI_Challenge_2012_W.xes.gz') #Change

    parameters['prefix_file'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC_2012W_Prefixes_all.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR, 'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR, 'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p')
    parameters['args'] = os.path.join(MILESTONE_DIR, 'args.p')
    parameters['milestone'] = MILESTONE
    parameters['prefix_length'] = 'fixed'

    parameters['pref_sql'] = os.path.join(MY_WORKSPACE_DIR, 'prefix_extraction.sql')
    parameters['test_traces'] = os.path.join(MY_WORKSPACE_DIR, 'bpic12w_test.csv')
    parameters['prefixes_train'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC12W_Prefixes_train.csv')
    parameters['prefixes_test'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC12W_Prefixes_test.csv')
    parameters['suff_pred_res'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC12w_suffpred_results.csv')

    parameters['log_name'] = 'bpic12w'

    return parameters

def _params_bpic17(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET = None, ENRICH = False):

    parameters = dict()
    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")

    parameters['max_size'] = MAX_SIZE
    parameters['min_size'] = MIN_SIZE
    parameters['n_size'] = N_SIZE
    parameters['subset'] = SUBSET

    parameters['event_log'] = os.path.join(MY_WORKSPACE_DIR, 'BPI_Challenge_2017.xes.gz')
    parameters['prefix_file'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC_2017_Prefixes_all.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR, 'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR, 'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p')
    parameters['args'] = os.path.join(MILESTONE_DIR, 'args.p')
    parameters['milestone'] = MILESTONE
    parameters['prefix_length'] = 'fixed'

    parameters['pref_sql'] = os.path.join(MY_WORKSPACE_DIR, 'prefix_extraction.sql')
    parameters['test_traces'] = os.path.join(MY_WORKSPACE_DIR, 'bpic17_test.csv')
    parameters['prefixes_train'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC17_Prefixes_train.csv')
    parameters['prefixes_test'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC17_Prefixes_test.csv')
    parameters['suff_pred_res'] = os.path.join(MY_WORKSPACE_DIR, 'BPIC17_suffpred_results.csv')
    parameters['log_name'] = 'bpic17'

    return parameters

def _params_helpdesk(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET=None, ENRICH=False):

    parameters = dict()
    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")

    parameters['max_size'] = MAX_SIZE
    parameters['min_size'] = MIN_SIZE
    parameters['n_size'] = N_SIZE
    parameters['subset'] = SUBSET

    parameters['event_log'] = os.path.join(MY_WORKSPACE_DIR, 'helpdesk.csv')
    parameters['prefix_file'] = os.path.join(MY_WORKSPACE_DIR, 'Helpdesk_Prefixes_all.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR, 'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR, 'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p')
    parameters['args'] = os.path.join(MILESTONE_DIR, 'args.p')
    parameters['milestone'] = MILESTONE
    parameters['prefix_length'] = 'fixed'

    parameters['pref_sql'] = os.path.join(MY_WORKSPACE_DIR, 'prefix_extraction.sql')
    parameters['test_traces'] = os.path.join(MY_WORKSPACE_DIR, 'helpdesk_test.csv')
    parameters['prefixes_train'] = os.path.join(MY_WORKSPACE_DIR, 'Helpdesk_Prefixes_train.csv')
    parameters['prefixes_test'] = os.path.join(MY_WORKSPACE_DIR, 'Helpdesk_Prefixes_test.csv')
    parameters['suff_pred_res'] = os.path.join(MY_WORKSPACE_DIR, 'Helpdesk_suffpred_results.csv')

    parameters['log_name'] = 'helpdesk'

    return parameters

def _params_synthetic(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE):

    parameters = dict()
    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")

    parameters['max_size'] = MAX_SIZE
    parameters['min_size'] = MIN_SIZE
    parameters['n_size'] = N_SIZE

    parameters['event_log'] = os.path.join(MY_WORKSPACE_DIR, 'Synthetic.xes')
    parameters['prefix_file'] = os.path.join(MY_WORKSPACE_DIR, 'Synthetic_Prefixes_all.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR, 'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR, 'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p')
    parameters['args'] = os.path.join(MILESTONE_DIR, 'args.p')
    parameters['milestone'] = MILESTONE
    parameters['prefix_length'] = 'fixed'

    parameters['pref_sql'] = os.path.join(MY_WORKSPACE_DIR, 'prefix_extraction.sql')
    parameters['test_traces'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_test.csv')
    parameters['prefixes_train'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_Prefixes_train.csv')
    parameters['prefixes_test'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_Prefixes_test.csv')
    parameters['suff_pred_res'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_suffpred_results.csv')

    parameters['log_name'] = 'synthetic'

    return parameters

def _params_synthetic_pre(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE):

    parameters = dict()
    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")

    parameters['max_size'] = MAX_SIZE
    parameters['min_size'] = MIN_SIZE
    parameters['n_size'] = N_SIZE

    parameters['event_log'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_pre.xes')
    parameters['prefix_file'] = os.path.join(MY_WORKSPACE_DIR, 'Synthetic_Prefixes_all.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR, 'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR, 'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p')
    parameters['args'] = os.path.join(MILESTONE_DIR, 'args.p')
    parameters['milestone'] = MILESTONE
    parameters['prefix_length'] = 'fixed'

    parameters['pref_sql'] = os.path.join(MY_WORKSPACE_DIR, 'prefix_extraction.sql')
    parameters['test_traces'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_test.csv')
    parameters['prefixes_train'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_Prefixes_train.csv')
    parameters['prefixes_test'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_Prefixes_test.csv')
    parameters['suff_pred_res'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_suffpred_results.csv')

    parameters['log_name'] = 'synthetic_pre'

    return parameters

def _params_synthetic_post(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE):

    parameters = dict()
    parameters['folder'] = os.path.join(MILESTONE_DIR, "output_files")

    parameters['max_size'] = MAX_SIZE
    parameters['min_size'] = MIN_SIZE
    parameters['n_size'] = N_SIZE

    parameters['event_log'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_post.xes')
    parameters['prefix_file'] = os.path.join(MY_WORKSPACE_DIR, 'Synthetic_Prefixes_all.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR, 'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR, 'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p')
    parameters['args'] = os.path.join(MILESTONE_DIR, 'args.p')
    parameters['milestone'] = MILESTONE
    parameters['prefix_length'] = 'fixed'

    parameters['pref_sql'] = os.path.join(MY_WORKSPACE_DIR, 'prefix_extraction.sql')
    parameters['test_traces'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_test.csv')
    parameters['prefixes_train'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_Prefixes_train.csv')
    parameters['prefixes_test'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_Prefixes_test.csv')
    parameters['suff_pred_res'] = os.path.join(MY_WORKSPACE_DIR, 'synthetic_suffpred_results.csv')

    parameters['log_name'] = 'synthetic_post'

    return parameters

def get_parameters(dataset, MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET=None, ENRICH=False):

    # elif dataset == 'bpic17':
    #     return _params_bpic17(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE,N_SIZE)
    # elif dataset == 'helpdesk':
    if dataset == 'bpic12':
        return _params_bpic12(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE,N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET, ENRICH)
    if dataset == 'bpic12w':
        return _params_bpic12W(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE,N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET, ENRICH)
    if dataset == 'helpdesk':
        return _params_helpdesk(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET, ENRICH)
    if dataset == 'bpic17':
        return _params_bpic17(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE, SUBSET, ENRICH)
    if dataset == 'synthetic':
        return _params_synthetic(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE)
    if dataset == 'synthetic_pre':
        return _params_synthetic_pre(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE)
    if dataset == 'synthetic_post':
        return _params_synthetic_post(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE)
    else:
        raise  ValueError("Please specific dataset 'bpic12', 'bpic12w', 'helpdesk' or 'bpic17'")