import os
import pickle
import shutil
import tempfile
import math
import cv2
import glob
import subprocess
import numpy as np
from collections import deque
from skimage import transform as tf
from tqdm import tqdm


# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform(
        "similarity", src, dst
    )  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped


# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception("too much bias in height")
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    cutted_img = np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )
    return cutted_img


def write_video_ffmpeg(rois, target_path, ffmpeg):
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    print(tmp_dir)
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals) + ".png"), roi)
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [
        ffmpeg,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(tmp_dir),
        "-q:v",
        "1",
        "-r",
        str(fps),
        "-y",
        "-crf",
        "20",
        target_path,
    ]
    subprocess.run(cmd)
    return


def crop_patch(
    frame_data,  # 비디오 프레임 데이터
    landmarks,
    mean_face_landmarks,
    stablePntsIDs,
    STD_SIZE,
    start_idx,
    stop_idx,
    crop_height,
    crop_width,
):
    """Crop mouth patch
    :param list frame_data: 비디오 프레임 배열
    :param list landmarks: 인터폴레이션된 랜드마크
    """
    sequence = []
    q_frame = deque()
    q_landmarks = deque()

    for frame_idx, (frame, cur_landmarks) in enumerate(zip(frame_data, landmarks)):
        q_frame.append(frame)
        q_landmarks.append(cur_landmarks)

        if len(q_frame) >= 1:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_frame = q_frame.popleft()
            cur_landmarks = q_landmarks.popleft()

            # -- affine transformation
            trans_frame, trans = warp_img(
                smoothed_landmarks[stablePntsIDs, :],
                mean_face_landmarks[stablePntsIDs, :],
                cur_frame,
                STD_SIZE,
            )
            trans_landmarks = np.dot(
                np.hstack([cur_landmarks, np.ones((cur_landmarks.shape[0], 1))]),
                trans.params.T,
            )[:, :2]

            # -- crop mouth patch
            patch = cut_patch(
                trans_frame,
                trans_landmarks[start_idx:stop_idx],
                crop_height // 2,
                crop_width // 2,
            )
            sequence.append(patch)

    return np.array(sequence)


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(
                landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
            )
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [
            landmarks[valid_frames_idx[0]]
        ] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
            len(landmarks) - valid_frames_idx[-1]
        )
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks
