import argparse
import json
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm


def save_label_file(sentence_text, output_dir, base_filename, sentence_id):
    """
    텍스트를 라벨 파일로 저장합니다.
    Args:
        sentence_text: str, 저장할 텍스트
        output_dir: str, 라벨 파일을 저장할 디렉토리
        base_filename: str, 원본 파일 이름
        sentence_id: int, 문장의 ID
    """
    try:
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        label_filename = f"{os.path.splitext(base_filename)[0]}_{sentence_id}.txt"
        label_path = os.path.join(output_dir, label_filename)

        with open(label_path, "w", encoding="utf-8") as label_file:
            label_file.write(sentence_text)

    except Exception as e:
        print(f"Error saving label file for sentence ID {sentence_id}: {e}")


def save_video_segment(
    input_video_path, output_dir, base_filename, sentence_id, start_time, end_time
):
    """
    비디오 구간을 추출합니다.
    Args:
        input_video_path: str, 원본 비디오 파일 경로
        output_dir: str, 구간별 비디오 파일을 저장할 디렉토리
        base_filename: str, 원본 파일 이름
        sentence_id: int, 문장의 ID
        start_time: float, 구간 시작 시간 (초)
        end_time: float, 구간 종료 시간 (초)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        output_filename = f"{os.path.splitext(base_filename)[0]}_{sentence_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # MoviePy로 비디오 로드 및 구간 추출
        with VideoFileClip(input_video_path) as video:
            video_segment = video.subclip(start_time, end_time)
            video_segment.write_videofile(
                output_path, codec="libx264", audio_codec="aac"
            )

        print(f"Video segment saved: {output_path}")
    except Exception as e:
        print(f"Error saving video segment for sentence ID {sentence_id}: {e}")


def save_audio_segment(
    input_audio_path, output_dir, base_filename, sentence_id, start_time, end_time
):
    """
    오디오 구간을 추출합니다.
    Args:
        input_audio_path: str, 원본 오디오 파일 경로
        output_dir: str, 구간별 오디오 파일을 저장할 디렉토리
        base_filename: str, 원본 파일 이름
        sentence_id: int, 문장의 ID
        start_time: float, 구간 시작 시간 (초)
        end_time: float, 구간 종료 시간 (초)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        output_filename = f"{os.path.splitext(base_filename)[0]}_{sentence_id}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # pydub을 사용하여 오디오 로드 및 구간 추출
        audio = AudioSegment.from_file(input_audio_path)
        audio_segment = audio[start_time * 1000 : end_time * 1000]
        audio_segment.export(output_path, format="wav")

        print(f"Audio segment saved: {output_path}")
    except Exception as e:
        print(f"Error saving audio segment for sentence ID {sentence_id}: {e}")


def process_file(
    json_file_path,
    label_output_dir,
    video_folder,
    video_output_dir,
    audio_folder,
    audio_output_dir,
):
    """
    단일 JSON 파일을 처리하여 텍스트 정보를 라벨 파일로 저장합니다.
    Args:
        json_file_path: str, JSON 파일 경로
        label_output_dir: str, 라벨 파일 저장 경로
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # JSON 구조 확인
        if not isinstance(data, list):
            raise ValueError(f"Unexpected JSON structure in file: {json_file_path}")

        for item in data:
            video_info = item.get("Video_info", {})
            audio_info = item.get("Audio_info", {})
            sentence_info = item.get("Sentence_info", [])

            video_name = video_info.get("video_Name", "Unknown")
            video_file_path = os.path.join(video_folder, video_name)

            audio_name = audio_info.get("Audio_Name", "Unknown")
            audio_file_path = os.path.join(audio_folder, audio_name)

            # Json 정보 추출
            print(f"Processing : {video_name.split('.')[0]}")
            for sentence in tqdm(sentence_info):
                sentence_id = sentence.get("ID")
                sentence_text = sentence.get("sentence_text")
                start_time = sentence.get("start_time")
                end_time = sentence.get("end_time")

                if None in (sentence_id, sentence_text, start_time, end_time):
                    print(f"Skipping incomplete sentence data: {sentence}")
                    continue

                # 라벨 파일 저장
                base_filename = os.path.basename(json_file_path)
                save_label_file(
                    sentence_text, label_output_dir, base_filename, sentence_id
                )

                # 비디오 구간 추출
                save_video_segment(
                    video_file_path,
                    video_output_dir,
                    base_filename,
                    sentence_id,
                    start_time,
                    end_time,
                )

                # 오디오 구간 추출
                save_audio_segment(
                    audio_file_path,
                    audio_output_dir,
                    base_filename,
                    sentence_id,
                    start_time,
                    end_time,
                )

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {json_file_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error while processing {json_file_path}: {e}")


def process_all_files(
    json_folder=None,
    label_output_dir=None,
    json_file=None,
    video_folder=None,
    video_output_dir=None,
    audio_folder=None,
    audio_output_dir=None,
):
    """
    JSON 파일들을 처리합니다.
    Args:
        json_folder: str, JSON 파일이 저장된 폴더 경로
        label_output_dir: str, 라벨 파일 저장 경로
        json_file: str, 처리할 특정 JSON 파일 이름 (옵션)
    """
    try:
        if json_file:
            # 특정 JSON 파일 처리
            json_file_path = os.path.join(json_folder, json_file)
            if not os.path.isfile(json_file_path):
                print(f"Error: File not found - {json_file_path}")
                return
            process_file(
                json_file_path,
                label_output_dir,
                video_folder,
                video_output_dir,
                audio_folder,
                audio_output_dir,
            )
        else:
            # 폴더 내 모든 JSON 파일 처리
            json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
            if not json_files:
                print(f"No JSON files found in folder: {json_folder}")
                return

            for json_file in tqdm(json_files, desc="Processing JSON files"):
                json_file_path = os.path.join(json_folder, json_file)
                process_file(
                    json_file_path,
                    label_output_dir,
                    video_folder,
                    video_output_dir,
                    audio_folder,
                    audio_output_dir,
                )

    except Exception as e:
        print(f"Error while processing JSON files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Raw Data.")
    # json 파일 처리
    parser.add_argument(
        "--json_folder", required=True, help="JSON 파일이 저장된 폴더 경로"
    )
    parser.add_argument("--json_file", help="처리할 특정 JSON 파일 이름 (옵션)")
    parser.add_argument(
        "--label_output_dir", required=True, help="라벨 파일을 저장할 디렉토리 경로"
    )
    # video 파일 처리
    parser.add_argument(
        "--video_folder", required=True, help="Video 파일이 저장된 폴더 경로"
    )
    parser.add_argument(
        "--video_output_dir",
        required=True,
        help="구간 분할된 영상 저장할 디렉토리 경로",
    )
    # audio 파일 처리
    parser.add_argument(
        "--audio_folder", required=True, help="Audio 파일이 저장된 폴더 경로"
    )
    parser.add_argument(
        "--audio_output_dir",
        required=True,
        help="구간 분할된 음성 저장할 디렉토리 경로",
    )

    args = parser.parse_args()

    process_all_files(
        json_folder=args.json_folder,
        label_output_dir=args.label_output_dir,
        json_file=args.json_file,
        video_folder=args.video_folder,
        video_output_dir=args.video_output_dir,
        audio_folder=args.audio_folder,
        audio_output_dir=args.audio_output_dir,
    )
