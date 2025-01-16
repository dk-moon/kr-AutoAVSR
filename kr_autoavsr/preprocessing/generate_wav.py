from moviepy.editor import VideoFileClip
import os
import argparse


# 비디오에서 오디오를 추출하여 wav 파일로 저장하는 함수
def extract_audio_from_video(video_file_path, audio_file_path):
    """
    비디오 파일에서 오디오를 추출하여 wav 파일로 저장합니다.
    Args:
        video_file_path (str): 입력 비디오 파일 경로
        audio_file_path (str): 출력 오디오 파일 경로
    """
    # 비디오 파일 로드
    video = VideoFileClip(video_file_path)
    # 비디오의 오디오 부분을 wav 파일로 저장 (샘플링 속도: 16kHz)
    video.audio.write_audiofile(audio_file_path, fps=16000)


# 폴더 및 파일명을 기반으로 오디오 추출 작업을 수행하는 함수
def process_files(folder_path, file_name=None):
    """
    지정된 폴더와 파일에서 오디오를 추출합니다.
    Args:
        folder_path (str): 비디오 파일이 포함된 폴더 경로
        file_name (str, optional): 처리할 특정 비디오 파일 이름 (기본값: None)
    """
    # 폴더 존재 여부 확인
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # 출력 오디오 파일을 저장할 'wav' 폴더 생성 (없을 경우 생성)
    wav_folder = os.path.join(folder_path, "wav")
    if not os.path.isdir(wav_folder):
        os.mkdir(wav_folder)  # 'wav' 폴더 생성
        print(f"'wav' 폴더가 {folder_path}에 생성되었습니다.")

    if file_name:
        # 특정 파일 처리
        video_file_path = os.path.join(folder_path, file_name)
        # 지정된 파일이 존재하지 않거나 .mp4 확장자가 아니면 오류 메시지 출력
        if not os.path.isfile(video_file_path) or ".mp4" not in file_name:
            print(f"Error: File '{file_name}' not found or not a valid .mp4 file.")
            return

        # 출력 오디오 파일 경로 설정
        audio_file_path = os.path.join(wav_folder, file_name.split(".")[0] + ".wav")
        # 오디오 추출 함수 호출
        extract_audio_from_video(video_file_path, audio_file_path)
        print(f"오디오 추출 완료: {file_name}")
    else:
        # 폴더 내 모든 파일 처리
        for file in os.listdir(folder_path):
            # 파일 확장자가 .mp4가 아니면 건너뜀
            if not file.endswith(".mp4"):
                continue
            # 비디오 파일 경로 설정
            video_file_path = os.path.join(folder_path, file)
            # 출력 오디오 파일 경로 설정
            audio_file_path = os.path.join(wav_folder, file.split(".")[0] + ".wav")
            # 오디오 추출 함수 호출
            extract_audio_from_video(video_file_path, audio_file_path)
            print(f"오디오 추출 완료: {file}")


# 메인 실행 부분
if __name__ == "__main__":
    """
    명령줄 인자를 통해 폴더 경로와 파일명을 입력받아 비디오에서 오디오를 추출합니다.
    """
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir",
        default="/home/aiv-gpu-019/data-all",
        help="비디오 파일이 포함된 폴더 경로",
    )
    parser.add_argument(
        "--test_fn",
        default="",
        help="처리할 특정 비디오 파일 이름 (비워두면 모든 파일 처리)",
    )
    args = parser.parse_args()

    # 지정된 폴더와 파일 이름을 기반으로 작업 수행
    process_files(args.test_dir, args.test_fn)
