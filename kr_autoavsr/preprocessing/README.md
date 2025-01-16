# 데이터 전처리

## Raw Data

- 구조
    ```
    .
    ├── audio
    │   ├── lip_J_1_F_02_C032_A_010.wav
    │   └── lip_J_1_F_02_C032_A_011.wav
    ├── json
    │   ├── lip_J_1_F_02_C032_A_010.json
    │   └── lip_J_1_F_02_C032_A_011.json
    └── video
        ├── lip_J_1_F_02_C032_A_010.mp4
        └── lip_J_1_F_02_C032_A_011.mp4
    ```

- Flow
    1. 데이터 준비
    2. 음성 데이터가 없을 경우 영상에서 음성 데이터 추출
        ```cmd
        python preprocessing/extract_wav.py \
        --video_dir {mp4 파일이 저장되어진 폴더}
        --video_fn {mp4 파일 명}
        ```
        - video_fn은 옵션으로 입력하지 않을 경우, video_dir의 모든 video에 대하여 작업 수행
    3. json 파일 정보 탐색
    4. json에 저장되어진 Sentence 정보 기준으로 대본 지문, 영상 및 음성 구간별 파일 추출
    5. sentencepiece를 이용하여 텍스트 토큰화 수행