import os
import argparse
import numpy

numpy.float = numpy.float64
numpy.int = numpy.int_
import numpy as np
import dlib
import cv2
import skvideo.io
from tqdm import tqdm
from align_mouth import landmarks_interpolate, crop_patch


def detect_landmark(image, detector, predictor):
    """Detect 68 facial landmarks in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for _, rect in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def video_generator(filename, size, fps, img_array):
    """Generate video from frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc=fourcc, fps=fps, frameSize=size)
    for img in img_array:
        out.write(img)
    out.release()


def preprocess_video(
    input_video_path, output_video_path, face_predictor_path, mean_face_path
):
    """Preprocess video to extract mouth region."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]

    # Read video frames
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []

    for frame in tqdm(frames, desc="Detecting landmarks"):
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)

    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(
        input_video_path,
        preprocessed_landmarks,
        mean_face_landmarks,
        stablePntsIDs,
        STD_SIZE,
        window_margin=12,
        start_idx=48,
        stop_idx=68,
        crop_height=96,
        crop_width=96,
    )

    # Generate video from ROIs
    size = (rois.shape[1], rois.shape[2])
    video_generator(output_video_path, size, 25, rois)
    print(f"Mouth video saved to: {output_video_path}")


def process_all_videos(video_dir, video_fn, face_predictor_path, mean_face_path):
    """
    Process videos in a directory or a specific video file.
    Args:
        video_dir: Directory containing videos.
        video_fn: Specific video file name (optional).
        face_predictor_path: Path to face landmark predictor.
        mean_face_path: Path to mean face landmarks.
    """
    if video_fn:
        # Process a single video
        video_path = os.path.join(video_dir, video_fn)
        if not os.path.exists(video_path) or not video_path.endswith(".mp4"):
            print(f"Error: {video_fn} is not a valid .mp4 file.")
            return

        output_dir = os.path.join(video_dir, "mouth")
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, video_fn)

        preprocess_video(
            video_path, output_video_path, face_predictor_path, mean_face_path
        )
    else:
        # Process all videos in the directory
        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        if not video_files:
            print(f"No .mp4 files found in directory: {video_dir}")
            return

        output_dir = os.path.join(video_dir, "mouth")
        os.makedirs(output_dir, exist_ok=True)

        for video_file in tqdm(video_files, desc="Processing all videos"):
            video_path = os.path.join(video_dir, video_file)
            output_video_path = os.path.join(output_dir, video_file)
            preprocess_video(
                video_path, output_video_path, face_predictor_path, mean_face_path
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos to extract mouth region."
    )
    parser.add_argument(
        "--video_dir", required=True, help="Directory containing .mp4 video files"
    )
    parser.add_argument("--video_fn", help="Specific .mp4 file to process (optional)")
    parser.add_argument(
        "--face_predictor_path", required=True, help="Path to dlib shape predictor file"
    )
    parser.add_argument(
        "--mean_face_path", required=True, help="Path to mean face landmarks file"
    )
    args = parser.parse_args()

    process_all_videos(
        video_dir=args.video_dir,
        video_fn=args.video_fn,
        face_predictor_path=args.face_predictor_path,
        mean_face_path=args.mean_face_path,
    )
