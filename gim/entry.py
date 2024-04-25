from DiffExtractor import DiffExtractor
import tqdm


def entry(extractor: DiffExtractor, t1_paths, t2_paths, mask_paths=None, threshold=0.1, size_ratio=1.0,
          warp_img=False, match_img=False):
    output = []
    for i in range(len(t1_paths)):
        img_a = t1_paths[i]
        img_b = t2_paths[i]
        mask = None
        if mask_paths is not None:
            mask = mask_paths[i]
        diff = extractor.extract(img_a, img_b, threshold=threshold, mask_path=mask, size_ratio=size_ratio,
                                 follow_up=warp_img or match_img)
        o = {'diff': diff}
        if warp_img:
            result = extractor.warp()
            o['warped_image'] = result
        if match_img:
            result = extractor.match()
            o['match_image'] = result

        output.append(o)
    return output


if __name__ == "__main__":
    import os
    import cv2 as cv

    extractor = DiffExtractor(device='cuda')
    t1_root = r'D:\models\north_campus\A'
    t2_root = r'D:\models\north_campus\B'
    t1_names = os.listdir(t1_root)
    t2_names = os.listdir(t2_root)
    t1_paths = [os.path.join(t1_root, f) for f in t1_names]
    t2_paths = [os.path.join(t2_root, f) for f in t2_names]

    output = 'output_north_small'
    os.makedirs(output, exist_ok=True)
    for i in tqdm.tqdm(range(0, len(t1_paths), 10)):
        out = entry(extractor, t1_paths[i:i + 10], t2_paths[i:i + 10], threshold=0.1, size_ratio=0.2, warp_img=True,
                    match_img=True)
        for j, o in enumerate(out):
            cv.imwrite(os.path.join(output, f'{t1_names[i + j]}_{t2_names[i + j]}_diff.png'), o['diff'])
            cv.imwrite(os.path.join(output, f'{t1_names[i + j]}_{t2_names[i + j]}_match.png'), o['match_image'])
            cv.imwrite(os.path.join(output, f'{t1_names[i + j]}_{t2_names[i + j]}_warped.png'), o['warped_image'])
