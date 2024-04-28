import math
import os
import sys
from enum import Enum

import numpy
import numpy as np
import cv2 as cv
import shutil

import torch

sys.path.append("nerfstudio")
sys.path.append("gim")

from flask import Blueprint, request, jsonify, make_response, url_for
from gim.DiffExtractor import DiffExtractor
from gim.entry import entry as gim_entry
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.scripts.train import main as train_main
from pathlib import Path
from nerfstudio.scripts.exporter import ExportNDFPointCloud
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, Model
from torchvision.ops import box_convert

use_dino=True
FP16_INFERENCE = True
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

main = Blueprint('main', __name__)

diff_extractor = DiffExtractor(checkpoints_path="gim/weights/gim_dkm_100h.ckpt", device='cuda')
model_dino = load_model(CONFIG_PATH, CHECKPOINT_PATH)


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
    output_path = ''
    output_matching_path = ''
    output_warped_path = ''

    def reset(self):
        self.process_stage = ProcessStage.start
        self.diff_extract_total_num = 0
        self.diff_extract_current_num = 0
        self.ndf_training_total_num = 0
        self.ndf_training_current_num = 0
        self.output_path = ''
        self.output_matching_path = ''
        self.output_warped_path = ''


process_state = ProcessState()


@main.route('/test', methods=['GET', 'POST'])
def test():
    test_data = request.args.get('test_data')
    if test_data:
        print(test_data)
    return jsonify({'test': 'success', 'test_url': url_for('static', filename='point_cloud.ply')})


@main.route('/start', methods=['GET', 'POST'])
def start():
    try:
        first_path = request.args.get('first_path')
        second_path = request.args.get('second_path')
        use_mask = bool(request.args.get('use_mask'))
        mask_path = request.args.get('mask_path')
        output_path = request.args.get('output_path')
        output_separately = bool(request.args.get('output_separately'))
        threshold = float(request.args.get('threshold'))
        size_ratio = float(request.args.get('size_ratio'))
        need_warp_img = bool(request.args.get('need_warp_img'))
        need_match_img = bool(request.args.get('need_match_img'))

        open_kernel_size = int(request.args.get('open_kernel_size'))
        close_kernel_size = int(request.args.get('close_kernel_size'))
        open_iter = int(request.args.get('open_iter'))
        close_iter = int(request.args.get('close_iter'))

        pose_file = request.args.get('pose_file')
        max_num_iterations = int(request.args.get('max_num_iterations'))
        export_point_num = int(request.args.get('export_point_num'))
        density_threshold = float(request.args.get('density_threshold'))
        rays_per_batch = int(request.args.get('rays_per_batch'))

        process_state.reset()

        if os.path.samefile(output_path, "tmp/"):
            shutil.rmtree(output_path)

        process_state.output_path = os.path.join(os.path.abspath(output_path), 'images')
        if output_separately:
            process_state.output_matching_path = os.path.join(os.path.abspath(output_path), 'match')
            process_state.output_warped_path = os.path.join(os.path.abspath(output_path), 'warped')

        if not use_mask:
            mask_path = None
        diff_extract(first_path, second_path, mask_path, output_path, output_separately, threshold, size_ratio,
                     need_warp_img, need_match_img, open_kernel_size, close_kernel_size, open_iter, close_iter)
        shutil.copy(pose_file, output_path)
        pose_file_name = os.path.basename(pose_file)
        point_cloud_path = ndf_entry(os.path.join(output_path, pose_file_name), max_num_iterations, rays_per_batch,
                                     export_point_num, density_threshold, vis='')
        shutil.copy(point_cloud_path, './backend/static')
        process_state.process_stage = ProcessStage.done
    except Exception as e:
        print(e)
        process_state.process_stage = ProcessStage.error
        return make_response(str(e), 500)
    return jsonify(
        {'point_cloud_path': os.path.join('http://127.0.0.1:8765/static', os.path.basename(point_cloud_path))})


@main.route('/status', methods=['GET'])
def status():
    return jsonify({'process_stage': process_state.process_stage.value,
                    'diff_extract_total_num': process_state.diff_extract_total_num,
                    'diff_extract_current_num': process_state.diff_extract_current_num,
                    'ndf_training_total_num': process_state.ndf_training_total_num,
                    'ndf_training_current_num': process_state.ndf_training_current_num,
                    'output_path': process_state.output_path,
                    'output_matching_path': process_state.output_matching_path,
                    'output_warped_path': process_state.output_warped_path})


def diff_extract(first_path, second_path, mask_path, output_path, output_separately, threshold, size_ratio,
                 need_warp_img,
                 need_match_img, open_kernel_size, close_kernel_size, open_iter, close_iter):
    process_state.process_stage = ProcessStage.reading_data
    t1_names = os.listdir(first_path)
    t2_names = os.listdir(second_path)
    t1_paths = [os.path.join(first_path, f) for f in t1_names]
    t2_paths = [os.path.join(second_path, f) for f in t2_names]
    mask_paths = None
    if mask_path != None:
        mask_names = os.listdir(mask_path)
        mask_paths = [os.path.join(mask_path, f) for f in mask_names]
        assert len(t1_paths) == len(mask_paths), "The number of images in the two folders must be the same"

    assert len(t1_paths) == len(t2_paths), "The number of images in the two folders must be the same"

    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    if output_separately:
        os.makedirs(os.path.join(output_path, 'match'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'warped'), exist_ok=True)

    process_state.diff_extract_total_num = len(t1_paths)
    process_state.diff_extract_current_num = 0
    for i in range(0, len(t1_paths)):
        process_state.process_stage = ProcessStage.feature_extracting
        # out = gim_entry(diff_extractor, t1_paths[i:i + 5], t2_paths[i:i + 5], mask_paths[i:i + 5] if mask_paths else None, threshold=threshold,
        #                 size_ratio=size_ratio, warp_img=need_warp_img, match_img=need_match_img)
        diff = diff_extractor.extract(t1_paths[i], t2_paths[i], threshold=threshold,
                                      mask_path=mask_paths[i] if mask_paths else None,
                                      follow_up=need_match_img or need_warp_img)
        warped_image = None
        matched_image = None
        if need_warp_img:
            warped_image = diff_extractor.warp()
        if need_match_img:
            matched_image = diff_extractor.match()

        process_state.process_stage = ProcessStage.post_processing

        # grounding_dino detection, only keep the boxes that contain the target
        if use_dino:
            boxes = grounding_dino_entry(model_dino, t1_paths[i], size_ratio=size_ratio, target='Buildings')
            if len(boxes) != 0:
                diff_after_dino = np.zeros_like(diff)
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x1=int(np.clip(x1, 0, diff.shape[1]))
                    x2=int(np.clip(x2, 0, diff.shape[1]))
                    y1=int(np.clip(y1, 0, diff.shape[0]))
                    y2=int(np.clip(y2, 0, diff.shape[0]))
                    diff_after_dino[y1:y2, x1:x2] = diff[y1:y2, x1:x2]
                diff=diff_after_dino

        diff_post = cv.morphologyEx(diff, cv.MORPH_OPEN,
                                    np.ones((open_kernel_size, open_kernel_size), np.uint8),
                                    iterations=open_iter)
        diff_post = cv.morphologyEx(diff_post, cv.MORPH_CLOSE,
                                    np.ones((close_kernel_size, close_kernel_size), np.uint8),
                                    iterations=close_iter)

        if output_separately:
            cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i]}'), diff_post)
            if matched_image is not None:
                cv.imwrite(os.path.join(output_path, 'match', f'{t1_names[i]}_{t2_names[i]}_match.png'),
                           matched_image)
            if warped_image is not None:
                cv.imwrite(os.path.join(output_path, 'warped', f'{t1_names[i]}_{t2_names[i]}_warped.png'),
                           warped_image)
        else:
            cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i]}'), diff_post)
            if matched_image is not None:
                cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i]}_{t2_names[i]}_match.png'),
                           matched_image)
            if warped_image is not None:
                cv.imwrite(os.path.join(output_path, 'images', f'{t1_names[i]}_{t2_names[i]}_warped.png'),
                           warped_image)
        process_state.diff_extract_current_num = min(process_state.diff_extract_current_num + 1,
                                                     process_state.diff_extract_total_num)


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


def grounding_dino_entry(model, image_path, size_ratio=1, text_prompt='Buildings. Clouds. Grasses. Sky. Floor.',
                         target='Buildings'):
    image_pil, image = load_image(image_path, size_ratio)

    if FP16_INFERENCE:
        image = image.half()
    model = model.half()

    boxes, _, phrases = predict(model, image, text_prompt, box_threshold=0.35, text_threshold=0.25, device='cuda')

    h, w, _ = image_pil.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    return [box for box, phrase in zip(xyxy, phrases) if target.lower() in phrase]
