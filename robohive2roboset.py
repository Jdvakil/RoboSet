import h5py
from logger import Robo_logger as Trace
import os
#Take a h5 file with robohive format and save it as roboset
#TODO: @jdvakil add config/derived for datasets 
def robohive2roboset(rollout_path, output_dir=None, max_paths=1e6):
    """
    rollout_path : path of the h5 file
    output_dir   : directory to save the new roboset h5
    max_paths    : number of rollouts to convert 
    """
    
    #check if file exists
    if not os.path.isfile(rollout_path):
        raise TypeError("File doesn't exist") 
    obj = h5py.File(rollout_path, "r")
    #check if file is of h5 format 
    if not isinstance(obj, h5py.Group):
        raise TypeError("File type not supported")
    if 'env_infos' not in obj[list(enumerate(obj))[0][1]].keys():
        raise TypeError("Format is not RoboHive")
    if output_dir == None:
        output_dir = os.path.dirname(rollout_path)
    rollout_name = os.path.split(rollout_path)[-1]
    file_name, file_type = os.path.splitext(rollout_name)
    output_name = os.path.join(output_dir, (file_name + "_roboset"))
    trace = Trace('roboset')
    datum = {}
    derived = {}
    count = 0
    for trial, value in obj.items():
        trace.create_group(trial)
        datum = {
            "time"      : value['time'],
            "rgb_left"  : value['env_infos/obs_dict/rgb:left_cam:240x424:2d'],
            "rgb_right" : value['env_infos']['obs_dict']['rgb:right_cam:240x424:2d'],
            "rgb_top"   : value['env_infos']['obs_dict']['rgb:top_cam:240x424:2d'], 
            "rgb_wrist" : value['env_infos']['obs_dict']['rgb:Franka_wrist_cam:240x424:2d'],
            "d_left"    : value['env_infos']['obs_dict']['d:left_cam:240x424:2d'],
            "d_right"   : value['env_infos']['obs_dict']['d:right_cam:240x424:2d'],
            "d_top"     : value['env_infos']['obs_dict']['d:top_cam:240x424:2d'],
            "d_wrist"   : value['env_infos']['obs_dict']['d:Franka_wrist_cam:240x424:2d'],
            "qp_arm"    : value['env_infos']['obs_dict']['qp_arm'],
            "qp_ee"     : value['env_infos']['obs_dict']['qp_ee'],
            "qv_arm"    : value['env_infos']['obs_dict']['qv_arm'],
            "qv_ee"     : value['env_infos']['obs_dict']['qv_ee'],
            "ctrl_arm"  : value['env_infos']['obs_dict']['ctrl_arm'],
            "ctrl_ee"   : value['env_infos']['obs_dict']['ctrl_ee'],
        }

        if 'pos_ee' in value['env_infos/obs_dict'].keys():
            derived['pos_ee'] = value["env_infos/obs_dict/pos_ee"]
        if 'rot_ee' in value['env_infos/obs_dict'].keys():
            derived['rot_ee'] = value["env_infos/obs_dict/rot_ee"]

        trace.append_datum_post_process(group_key=trial, dataset_key='derived', dataset_val=derived)
        trace.append_datum_post_process(group_key=f'{trial}', dataset_key='data', dataset_val=datum)
        
        if 'user_cmt' in value.keys():
            for _, comment in enumerate(value['user_cmt']):
                trace.create_dataset(group_key=trial, dataset_key='config/solved', dataset_val=np.float16(comment))
        count = count + 1
        if count >= max_paths:
            break
    trace.flatten()
    trace.save(trace_name=f"{output_name}.h5")