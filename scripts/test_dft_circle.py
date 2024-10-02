from im_test.dft_circle import DFTMarker
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
import albumentations as A
from im_test.augmentations import Rotate90, CropRatio, Identity
import numpy as np
import cv2
import traceback
aug_list = [
    ('identity', Identity()),
    ('jpeg75', A.ImageCompression(75, 75, always_apply=True)),
    ('jpeg50', A.ImageCompression(50, 50, always_apply=True)),
    ('jpeg20', A.ImageCompression(20, 20, always_apply=True)),
    ('rotate90', Rotate90()),
    ('rotate30', A.Rotate(limit=(30, 30), always_apply=True)),
    ('rotate60', A.Rotate(limit=(60, 60), always_apply=True)),
    ('gauss_blur_3', A.GaussianBlur((3, 3), always_apply=True)),
    ('gaus_blur_5', A.GaussianBlur((5, 5), always_apply=True)),
    ('gaus_blur_7', A.GaussianBlur((7, 7), always_apply=True)),
    ('gaus_noise_8', A.GaussNoise((8, 8), always_apply=True)),
    ('gaus_noise_13', A.GaussNoise((13, 13), always_apply=True)),
    ('gaus_noise_22', A.GaussNoise((22, 22), always_apply=True)),
    ('center_crop_80', CropRatio(0.8)),
    ('center_crop_50', CropRatio(0.5)),
    ('center_crop_30', CropRatio(0.3)),
]
rnd_mark = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
                     0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                     0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                     1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
                     0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
                     1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1,
                     0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
                     1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                     1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
                     0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
                     1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
                     1, 0, 0, 0, 1, 1, 0, 0])


def mp_func(img, marker: DFTMarker, augmentations, marker_params):
    result_obj = {}
    mark = marker_params['mark']
    alpha = marker_params['alpha']
    try:
        img_emb = marker.embed(img, mark, alpha)
    except Exception as e:
        result_obj['embedded'] = False
        return result_obj
    psnr_metric = psnr(img, img_emb)
    result_obj['embedded'] = True
    result_obj['psnr'] = psnr_metric
    result_obj['extracted'] = []
    for aug_name, aug_obj in augmentations:
        aug_img = aug_obj(image=img)
        aug_img = aug_img['image']
        try:
            max_cor = marker.extract(aug_img, mark)
        except Exception as e:
            traceback.print_exc()
            result_obj['extracted'].append(None)
        else:
            result_obj['extracted'].append(max_cor)
    return result_obj


def mp_wrapper(args):
    return mp_func(*args)


def test_algorithm(algorithm, dataset_path, algorithm_name, dataset_name, res_dir,):
    test_rep_name = f'{dataset_name}__{algorithm_name}'
    res_rep_dir = Path(res_dir) / test_rep_name
    jsons_dir = Path(res_rep_dir) / 'jsons'
    jsons_dir.mkdir(exist_ok=True, parents=True)
    meta_file_path = res_rep_dir / 'meta.json'

    res_rep_dir.mkdir(exist_ok=True)
    augs = [i for i, _ in aug_list]
    meta = {'augmentations': augs}
    with open(meta_file_path, 'w') as f:
        json.dump(meta, f)

    img_path_list = sorted(os.listdir(dataset_path))
    step = 8
    marker_params = {'mark': rnd_mark, 'alpha': 600}
    for i in range(0, len(img_path_list), step):
        print(f'Processing {i}-{i+step}:')
        sublist = img_path_list[i:i + step]
        args = []
        for name in sublist:
            img = cv2.imread(Path(dataset_path) / name)
            args.append([img, algorithm, aug_list, marker_params])
        with ProcessPoolExecutor(step) as pool:
            result = pool.map(mp_wrapper, args)
        for ind, res in enumerate(result):
            res_path = jsons_dir / f'{sublist[ind]}.json'
            with open(res_path, 'w') as f:
                json.dump(res, f)


def main():
    marker = DFTMarker()
    ds_path = '/home/gtp/projects/yamark/filtered'
    res_dir = '/home/gtp/projects/yamark/test_results'
    test_algorithm(marker, ds_path, 'identity_extract',
                   'diffusiondb_filtered', res_dir)


if __name__ == '__main__':
    main()
