import os
import cv2
import sentencepiece as spm
import torch
from tqdm import tqdm
import argparse


class TextTransform:
    """SentencePiece를 이용한 텍스트 처리 클래스."""

    def __init__(self, sp_model_path, dict_path):
        # SentencePiece 모델 로드
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        # 유닛 파일 로드 및 해시맵 생성
        units = open(dict_path, encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}

        # <blank>, <eos> 토큰 추가
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        """텍스트를 SentencePiece 토큰으로 변환."""
        tokens = self.sp.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))


def process_data(
    mouth_dir,
    label_dir,
    audio_dir,
    csv_output_dir,
    txt_output_dir,
    spm_model_path,
    spm_dict_path,
):
    """
    mouth, label, audio 데이터를 처리하여 CSV와 TXT 파일로 저장.
    Args:
        mouth_dir: str, mouth 데이터 경로
        label_dir: str, label 데이터 경로
        audio_dir: str, audio 데이터 경로
        csv_output_dir: str, CSV 파일 저장 경로
        txt_output_dir: str, TXT 파일 저장 경로
        spm_model_path: str, SentencePiece 모델 경로
        spm_dict_path: str, SentencePiece 유닛 파일 경로
    """
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(txt_output_dir, exist_ok=True)

    # SentencePiece 모델 초기화
    text_transform = TextTransform(
        sp_model_path=spm_model_path, dict_path=spm_dict_path
    )

    # Label 파일 기준으로 동일한 파일명 검색
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    for label_file in tqdm(label_files, desc="Processing data"):
        file_base = os.path.splitext(label_file)[0]

        # 각 파일 경로 확인
        mouth_file = os.path.join(mouth_dir, f"{file_base}.mp4")
        label_file_path = os.path.join(label_dir, label_file)
        audio_file = os.path.join(audio_dir, f"{file_base}.wav")

        if not os.path.exists(mouth_file):
            print(f"Missing mouth file for {file_base}, skipping.")
            continue
        if not os.path.exists(audio_file):
            print(f"Missing audio file for {file_base}, skipping.")
            continue

        # 라벨 읽기
        with open(label_file_path, "r", encoding="utf-8") as label_f:
            label_text = label_f.readline().strip()

        # 비디오 프레임 수 추출
        video = cv2.VideoCapture(mouth_file)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # SentencePiece 토큰화
        token_ids = text_transform.tokenize(label_text)
        token_id_str = " ".join(map(str, [_.item() for _ in token_ids]))

        # CSV 저장
        csv_file_path = os.path.join(csv_output_dir, f"{file_base}.csv")
        with open(csv_file_path, "w", encoding="utf-8") as csv_f:
            csv_f.write(f"dataset,video_file,vframes,token_ids\n")
            csv_f.write(f"{mouth_dir},{file_base}.mp4,{frame_count},{token_id_str}\n")

        # # TXT 저장
        # txt_file_path = os.path.join(txt_output_dir, f"{file_base}.txt")
        # with open(txt_file_path, "w", encoding="utf-8") as txt_f:
        #     txt_f.write(label_text)

        print(f"Processed {file_base}: CSV saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process mouth, label, and audio data."
    )
    parser.add_argument(
        "--mouth_dir", required=True, help="mouth 데이터가 저장된 디렉토리 경로"
    )
    parser.add_argument(
        "--label_dir", required=True, help="label 데이터가 저장된 디렉토리 경로"
    )
    parser.add_argument(
        "--audio_dir", required=True, help="audio 데이터가 저장된 디렉토리 경로"
    )
    parser.add_argument(
        "--csv_output_dir", required=True, help="CSV 파일 저장 디렉토리 경로"
    )
    parser.add_argument(
        "--txt_output_dir", required=True, help="TXT 파일 저장 디렉토리 경로"
    )
    parser.add_argument(
        "--spm_model_path", required=True, help="SentencePiece 모델 경로"
    )
    parser.add_argument(
        "--spm_dict_path", required=True, help="SentencePiece 유닛 파일 경로"
    )

    args = parser.parse_args()

    process_data(
        mouth_dir=args.mouth_dir,
        label_dir=args.label_dir,
        audio_dir=args.audio_dir,
        csv_output_dir=args.csv_output_dir,
        txt_output_dir=args.txt_output_dir,
        spm_model_path=args.spm_model_path,
        spm_dict_path=args.spm_dict_path,
    )
