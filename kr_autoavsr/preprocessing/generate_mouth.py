import os
import argparse
import numpy

numpy.float = numpy.float64
numpy.int = numpy.int_
import numpy as np
import dlib
import cv2
from tqdm import tqdm
import skvideo.io
from align_mouth import landmarks_interpolate, crop_patch


def detect_landmark(image, detector, predictor):
    """
    주어진 이미지에서 얼굴 랜드마크를 탐지합니다.
    Args:
        image (numpy array): 입력 이미지
        detector (dlib.get_frontal_face_detector): 얼굴 탐지기
        predictor (dlib.shape_predictor): 랜드마크 예측기
    Returns:
        coords (numpy array): 68개 랜드마크 좌표
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            return None
        for rect in rects:
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            return coords
    except Exception as e:
        print(f"Error detecting landmarks: {e}")
        return None


def video_generator(filename, size, fps, img_array):
    """
    이미지 배열로부터 비디오 파일을 생성합니다.
    Args:
        filename (str): 출력 비디오 파일 경로
        size (tuple): 비디오 프레임 크기
        fps (int): 초당 프레임 수
        img_array (numpy array): 이미지 배열
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc=fourcc, fps=fps, frameSize=size)

        for img in img_array:
            out.write(img)

        out.release()
    except Exception as e:
        print(f"Error generating video: {e}")


def preprocess_video(
    input_video_path,
    output_video_path,
    face_predictor_path,
    mean_face_path,
    batch_size=100,
):
    """
    skvideo.io를 사용해 비디오 파일을 읽고 프레임 단위로 배치 처리하여 ROI를 추출합니다.
    """
    # 랜드마크 탐지기 및 예측기 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)

    # 평균 얼굴 랜드마크 로드
    if not os.path.isfile(mean_face_path):
        print(f"Error: Mean face landmarks file not found - {mean_face_path}")
        return
    mean_face_landmarks = np.load(mean_face_path, allow_pickle=True)

    STD_SIZE = (256, 256)
    stablePntsIDs = [33, 36, 39, 42, 45]

    frames = []
    landmarks = []
    rois = []

    try:
        # skvideo.io.vreader를 사용해 비디오 프레임 읽기
        videogen = skvideo.io.vreader(input_video_path)

        for frame_idx, frame in enumerate(tqdm(videogen, desc="Processing frames")):
            if frame is None or frame.size == 0:
                print(
                    f"Warning: Empty or invalid frame at index {frame_idx}. Skipping."
                )
                continue

            # RGB 형식에서 랜드마크 탐지
            landmark = detect_landmark(frame, detector, predictor)
            frames.append(frame)
            if landmark is not None:
                landmarks.append(landmark)
            else:
                landmarks.append(None)  # 랜드마크가 감지되지 않은 프레임

            # 배치 크기만큼 프레임이 쌓이면 처리
            if len(frames) == batch_size:
                process_batch(
                    frames,
                    landmarks,
                    mean_face_landmarks,
                    rois,
                    STD_SIZE,
                    stablePntsIDs,
                )
                frames.clear()
                landmarks.clear()

        # 남은 프레임 처리
        if frames:
            process_batch(
                frames, landmarks, mean_face_landmarks, rois, STD_SIZE, stablePntsIDs
            )

    except Exception as e:
        print(f"Error reading video with skvideo.io: {e}")
        return

    # ROI가 없으면 처리 중단
    if not rois:
        print("No valid ROIs extracted. Skipping video.")
        return

    # 비디오 생성 및 저장
    size = (rois[0].shape[1], rois[0].shape[2])
    video_generator(output_video_path, size, 25, np.array(rois, dtype=np.uint8))
    print(f"Processed video saved at: {output_video_path}")


def process_batch(
    frames, landmarks, mean_face_landmarks, rois, STD_SIZE, stablePntsIDs
):
    """
    배치 단위로 프레임과 랜드마크를 처리해 ROI를 추출합니다.
    """
    try:
        # 랜드마크 보간
        interpolated_landmarks = landmarks_interpolate(landmarks)
        if interpolated_landmarks is None:
            print("No valid landmarks after interpolation for batch. Skipping batch.")
            return

        # ROI 추출
        batch_rois = crop_patch(
            frames,
            interpolated_landmarks,
            mean_face_landmarks,
            stablePntsIDs,
            STD_SIZE,
            start_idx=48,
            stop_idx=68,
            crop_height=96,
            crop_width=96,
        )
        rois.extend(batch_rois)
    except Exception as e:
        print(f"Error during batch processing: {e}")


def process_files(folder_path, file_name, face_predictor_path, mean_face_path):
    """
    지정된 폴더와 파일명을 기반으로 입 모양 비디오를 생성합니다.
    """
    mouth_folder = os.path.join(folder_path, "mouth")
    os.makedirs(mouth_folder, exist_ok=True)

    if file_name:
        video_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(video_path) or not file_name.endswith(".mp4"):
            print(f"Error: {file_name} is not a valid .mp4 file.")
            return
        output_video_path = os.path.join(mouth_folder, file_name)
        preprocess_video(
            video_path, output_video_path, face_predictor_path, mean_face_path
        )
    else:
        for file in os.listdir(folder_path):
            if not file.endswith(".mp4"):
                continue
            video_path = os.path.join(folder_path, file)
            output_video_path = os.path.join(mouth_folder, file)
            preprocess_video(
                video_path, output_video_path, face_predictor_path, mean_face_path
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir",
        default="./tests/sample/video",
        help="비디오 파일이 포함된 폴더 경로",
    )
    parser.add_argument(
        "--test_fn",
        default="",
        help="처리할 특정 비디오 파일 이름 (비워두면 모든 파일 처리)",
    )
    parser.add_argument(
        "--face_predictor_path",
        default="./kr_autoavsr/preprocessing/shape_predictor_68_face_landmarks.dat",
        help="랜드마크 모델 경로",
    )
    parser.add_argument(
        "--mean_face_path",
        default="./kr_autoavsr/preprocessing/20words_mean_face.npy",
        help="평균 얼굴 랜드마크 경로",
    )
    args = parser.parse_args()

    process_files(
        args.test_dir, args.test_fn, args.face_predictor_path, args.mean_face_path
    )
