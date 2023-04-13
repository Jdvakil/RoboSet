from robohive.logger.grouped_datasets import Trace

class Robo_logger(Trace):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
    
    # duplicate of append_datum with h5 dataset copy vs list append.
    def append_datum_post_process(self, group_key, dataset_key, dataset_val):
        assert group_key in self.trace.keys(), "Group:{} does not exist".format(group_key)
        if dataset_key in self.trace[group_key].keys():
            self.verify_type(dataset=self.trace[group_key][dataset_key], data=dataset_val)
            self.trace[group_key][dataset_key] = (dataset_val)
        else:
            self.trace[group_key][dataset_key] = dataset_val
    