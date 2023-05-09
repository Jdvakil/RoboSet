import h5py
from logger import Robo_logger
import click
import ntpath
import os
import glob 
from robohive.utils.prompt_utils import prompt, Prompt

def splitter(pathname:str, even:str, odd:str):
    assert os.path.isdir(pathname), "Directory not found"
    files = glob.glob(pathname + "*.h5")
    for filename in files:
        prompt(filename, color="cyan", type=Prompt.INFO, flush=True)
        path = h5py.File(filename, 'r')

        action1_trace = Robo_logger('action1')
        action2_trace = Robo_logger("action2")
        action1 = {
        } #even trials
        action2 = {} #odd trials

        for trial, value in path.items():
            if int(trial[-1:]) % 2 == 0:
                action1['time'] = (value['data/time'])
                action1['rgb_left'] = value['data/rgb_left']
                action1['rgb_right'] = value['data/rgb_right']
                action1['rgb_top'] = value['data/rgb_top']
                action1['rgb_wrist'] = value['data/rgb_wrist']
                action1['d_left'] = value['data/d_left']
                action1['d_right'] = value['data/d_right']
                action1['d_top'] = value['data/d_top']
                action1['d_wrist'] = value['data/d_wrist']
                action1['qp_arm'] = value['data/qp_arm']
                action1['qp_ee'] = value['data/qp_ee']
                action1['qv_ee'] = value['data/qv_ee']
                action1['qv_arm'] = value['data/qv_arm']
                action1['ctrl_arm'] = value['data/ctrl_arm']
                action1['ctrl_ee'] = value['data/ctrl_ee']
                action1_trace.create_group(f'{trial}')
                action1_trace.append_datum_post_process(group_key=f'{trial}', dataset_key='data', dataset_val=action1)

            if int(trial[-1:]) % 2 != 0:
                action2[f'time'] = (value['data/time'])
                action2[f'rgb_left'] = value['data/rgb_left']
                action2[f'rgb_right'] = value['data/rgb_right']
                action2[f'rgb_top'] = value['data/rgb_top']
                action2[f'rgb_wrist'] = value['data/rgb_wrist']
                action2[f'd_left'] = value['data/d_left']
                action2[f'd_right'] = value['data/d_right']
                action2[f'd_top'] = value['data/d_top']
                action2[f'd_wrist'] = value['data/d_wrist']
                action2['qp_arm'] = value['data/qp_arm']
                action2['qp_ee'] = value['data/qp_ee']
                action2['qv_ee'] = value['data/qv_ee']
                action2['qv_arm'] = value['data/qv_arm']
                action2['ctrl_arm'] = value['data/ctrl_arm']
                action2['ctrl_ee'] = value['data/ctrl_ee']
                action2_trace.create_group(f'{trial}')
                action2_trace.append_datum_post_process(group_key=f'{trial}', dataset_key='data', dataset_val=action2)

        file = ntpath.basename(filename)[:-3]
        action1_trace.save(f'{file.replace(f"_{odd}", "")}.h5', verify_length=True)
        action2_trace.save(f'{file.replace(f"_{even}", "")}.h5', verify_length=True)

@click.command(help="Hi")
@click.option('-p', '--path', type=str)
@click.option('-e', '--even', type=str)
@click.option('-o', '--odd', type=str)
def main(path, even, odd):
    splitter(path, even, odd)

if __name__ == "__main__":
    main()
 