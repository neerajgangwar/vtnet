import os

from utils.data_utils import loading_scene_list
from utils.model_util import ScalarMeanTracker

os.environ["OMP_NUM_THREADS"] = "1"
import torch.multiprocessing as mp

import time
import json
from tqdm import tqdm

from runners import a3c_val


def main_eval(args, saved_model, outdir, device):
    scenes = loading_scene_list(args)
    processes = []

    res_queue = mp.Queue()
    args.learned_loss = False
    args.num_steps = 50
    target = a3c_val

    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                saved_model,
                res_queue,
                250,
                scene_type,
                scenes[rank],
                device,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)
    train_scalars = ScalarMeanTracker()

    visualizations = []

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)
            visualizations.append(train_result['tools'])
        
        tracked_means = train_scalars.pop_and_reset()
    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    with open(args.results_json, "w") as fp:
        json.dump(tracked_means, fp, sort_keys=True, indent=4)

    visualization_dir = f"{outdir}/visualization_files"
    if not os.path.exists(visualization_dir):
        os.mkdir(visualization_dir)

    with open(os.path.join(visualization_dir, args.visualize_file_name), 'w') as wf:
        json.dump(visualizations, wf)
