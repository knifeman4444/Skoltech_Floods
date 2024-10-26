'''
Для запуска необходимо создать виртуальное окружение из deepwatermap_requirements.txt
т.к. оригинальный код использует 2 версию keras.

Инструкция как запускать код со старым keras, если что-то пошло не так:
https://github.com/tensorflow/tensorflow/releases?q=keras+&expanded=true


Этот скрипт выполняет готовую модель на всех изображениях в папке, и сохраняет рядом вывод модели.
Далее результат работы модели можно подать как канал на вход другой модели, либо же просто сблендить результаты.
'''

# Uncomment this to run inference on CPU if your GPU runs out of memory
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tqdm import tqdm
import os;os.environ["TF_USE_LEGACY_KERAS"]="1"

import argparse
import deepwatermap
import tifffile as tiff
import numpy as np
import cv2


def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


def inference_for_image(model, image_dir, image_name):
    save_path = os.path.join(image_dir, "dwm_" + image_name + ".png")

    if os.path.exists(save_path):
        return

    image_path = os.path.join(image_dir, image_name + ".tif")

    # load and preprocess the input image
    image = tiff.imread(image_path)
    is0, is1, _ = image.shape
    pad_r = find_padding(image.shape[0])
    pad_c = find_padding(image.shape[1])
    image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')

    # solve no-pad index issue after inference
    if pad_r[1] == 0:
        pad_r = (pad_r[0], 1)
    if pad_c[1] == 0:
        pad_c = (pad_c[0], 1)

    image = image[:, :, [0, 1, 2, 6, 8, 9]]

    image = image.astype(np.float32)

    # remove nans (and infinity) - replace with 0s
    image = np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    image = image - np.min(image)
    image = image / np.maximum(np.max(image), 1)

    # run inference
    image = np.expand_dims(image, axis=0)
    dwm = model.predict(image)
    dwm = np.squeeze(dwm)
    dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]

    # soft threshold
    dwm = 1./(1+np.exp(-(16*(dwm-0.5))))
    dwm = np.clip(dwm, 0, 1)
    p0 = is0 - dwm.shape[0]
    p1 = is1 - dwm.shape[1]

    dwm = np.pad(dwm, ((0, p0), (0, p1)), mode='reflect')

    # save the output water map
    cv2.imwrite(save_path, dwm * 255)


def launch_inference(checkpoint_path, image_dir):
    # load the model
    model = deepwatermap.model()
    model.load_weights(checkpoint_path)

    for root, dirs, files in os.walk(image_dir):
        for file in tqdm(sorted(files)):
            if file.endswith('.tif'):
                image_name = file[:-4]
                inference_for_image(model, image_dir, image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        help="Path to the dir where the checkpoints are stored")
    parser.add_argument('--image_dir', type=str, help="Path to the input GeoTIFF image dir")
    args = parser.parse_args()
    launch_inference(args.checkpoint_path, args.image_dir)
