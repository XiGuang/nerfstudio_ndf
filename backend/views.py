import math
import os
from enum import Enum
import numpy as np
import cv2 as cv
import shutil

from flask import Blueprint, request, jsonify, make_response
from gim.DiffExtractor import DiffExtractor
from gim.entry import entry as gim_entry
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.scripts.train import main as train_main
from pathlib import Path
from nerfstudio.scripts.exporter import ExportNDFPointCloud

main = Blueprint('main', __name__)

diff_extractor = DiffExtractor(checkpoints_path="gim/weights/gim_dkm_100h.ckpt", device='cuda')


class ProcessStage(Enum):
    start = 0
    reading_data = 1
    feature_extracting = 2
    post_processing = 3
    ndf_training = 4
    ndf_exporting = 5
    done = 6
    error = 7


class ProcessState():
    process_stage = ProcessStage.start
    diff_extract_total_num = 0
    diff_extract_current_num = 0
    ndf_training_total_num = 0
    ndf_training_current_num = 0

    def reset(self):
        self.process_stage = ProcessStage.start
        self.diff_extract_total_num = 0
        self.diff_extract_current_num = 0
        self.ndf_training_total_num = 0
        self.ndf_training_current_num = 0


process_state = ProcessState()


@main.route('/start', methods=['GET'])
def start():
    try:
        first_path = request.args.get('first_path')
        second_path = request.args.get('second_path')
        use_mask = request.args.get('use_mask')
        mask_path = request.args.get('mask_path')
        output_path = request.args.get('output_path')
        output_separately = request.args.get('output_separately')
        threshold = request.args.get('threshold')
        size_ratio = request.args.get('size_ratio')
        need_warp_img = request.args.get('need_warp_img')
        need_match_img = request.args.get('need_match_img')

        open_kernel_size = request.args.get('open_kernel_size')
        close_kernel_size = request.args.get('close_kernel_size')
        open_iter = request.args.get('open_iter')
        close_iter = request.args.get('close_iter')

        pose_file = request.args.get('pose_file')
        max_num_iterations = request.args.get('max_num_iterations')
        export_point_num = request.args.get('export_point_num')
        density_threshold = request.args.get('density_threshold')
        rays_per_batch = request.args.get('rays_per_batch')

        process_state.reset()

        if not use_mask:
            mask_path = None
        diff_extract(first_path, second_path, mask_path, output_path, output_separately, threshold, size_ratio,
                     need_warp_img, need_match_img, open_kernel_size, close_kernel_size, open_iter, close_iter)
        shutil.copy(pose_file, output_path)
        pose_file_name = os.path.basename(pose_file)
        point_cloud_path = ndf_entry(os.path.join(output_path, pose_file_name), max_num_iterations, rays_per_batch,
                                     export_point_num, density_threshold, vis='')
        process_state.process_stage = ProcessStage.done
    except Exception as e:
        process_state.process_stage = ProcessStage.error
        return make_response(str(e), 500)
    return jsonify({'point_cloud_path': point_cloud_path})


def diff_extract(first_path, second_path, mask_path, output_path, output_separately, threshold, size_ratio,
                 need_warp_img,
                 need_match_img, open_kernel_size, close_kernel_size, open_iter, close_iter):
    process_state.process_stage = ProcessStage.reading_data
    t1_names = os.listdir(first_path)
    t2_names = os.listdir(second_path)
    t1_paths = [os.path.join(first_path, f) for f in t1_names]
    t2_paths = [os.path.join(second_path, f) for f in t2_names]
    mask_paths = None
    if not mask_path:
        mask_names = os.listdir(mask_path)
        mask_paths = [os.path.join(mask_path, f) for f in mask_names]

    assert len(t1_paths) == len(t2_paths), "The number of images in the two folders must be the same"
    assert len(t1_paths) == len(mask_paths), "The number of images in the two folders must be the same"

    if output_separately:
        os.makedirs(os.path.join(output_path, 'match'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'warped'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    process_state.diff_extract_total_num = math.ceil(len(t1_paths) / 5)
    process_state.diff_extract_current_num = 0
    for i in range(0, len(t1_paths), 5):
        process_state.process_stage = ProcessStage.feature_extracting
        out = gim_entry(diff_extractor, t1_paths[i:i + 5], t2_paths[i:i + 5], mask_paths[i:i + 5], threshold=threshold,
                        size_ratio=size_ratio, warp_img=need_warp_img, match_img=need_match_img)
        diffs = [o['diff'] for o in out]
        matches = [o['match_image'] for o in out]
        warps = [o['warped_image'] for o in out]

        process_state.process_stage = ProcessStage.post_processing

        diffs_post = []
        for diff in diffs:
            diff = cv.morphologyEx(diff, cv.MORPH_OPEN, np.ones((open_kernel_size, open_kernel_size), np.uint8),
                                   iterations=open_iter)
            diff = cv.morphologyEx(diff, cv.MORPH_CLOSE, np.ones((close_kernel_size, close_kernel_size), np.uint8),
                                   iterations=close_iter)
            diffs_post.append(diff)

        for j in range(len(diffs)):
            if output_separately:
                cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i + j]}'), diffs_post[j])
                if matches:
                    cv.imwrite(os.path.join(output_path, 'match', f'{t1_names[i + j]}_{t2_names[i + j]}_match.png'),
                               matches[j])
                if warps:
                    cv.imwrite(os.path.join(output_path, 'warped', f'{t1_names[i + j]}_{t2_names[i + j]}_warped.png'),
                               warps[j])
            else:
                cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i + j]}'), diffs_post[i])
                if matches:
                    cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i + j]}_{t2_names[i + j]}_match.png'),
                               matches[i])
                if warps:
                    cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i + j]}_{t2_names[i + j]}_warped.png'),
                               warps[i])
        process_state.diff_extract_current_num = max(i + 5, process_state.diff_extract_total_num)


def ndf_entry(data_path: str, max_num_iterations: int = 3000, train_num_rays_per_batch: int = 1024,
              num_points: int = 1000000, density_threshold: float = 0.9, output_dir: str = "outputs",
              vis: str = "viewer", ):
    config = method_configs['ndf']
    config.pipeline.model.background_color = "black"
    config.data = Path(data_path)
    config.max_num_iterations = max_num_iterations
    config.pipeline.datamanager.train_num_rays_per_batch = train_num_rays_per_batch
    config.vis = vis

    process_state.process_stage = ProcessStage.ndf_training
    train_main(config)

    process_state.process_stage = ProcessStage.ndf_exporting
    ExportNDFPointCloud(num_points=num_points, density_threshold=density_threshold,
                        load_config=Path(os.path.join(config.get_base_dir(), "config.yml")),
                        output_dir=Path(output_dir)).main()
    return os.path.join(output_dir, "point_cloud.ply")
