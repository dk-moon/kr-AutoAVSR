import os

import torch
import torchaudio
import torchvision
import soundfile as sf
import librosa

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    wav, sr = sf.read(path[:-4] + ".wav")
    if wav.ndim == 2:
        wav = wav.mean(-1)
    assert sr== 16_000 and len(wav.shape) == 1
    return torch.Tensor(wav).unsqueeze(1)

class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        mouth_dir,
        wav_dir,
        rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        self.list = self.load_list(label_path)

        self.audio_transform = audio_transform
        self.video_transform = video_transform

        self.mouth_dir = mouth_dir
        self.wav_dir = wav_dir

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    torch.tensor([int(_) for _ in token_id.split()]),
                )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if self.modality == "video":
            rel_path1 = os.path.join(self.mouth_dir,rel_path)
            path = os.path.join(self.root_dir,dataset_name, rel_path1)
     
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            rel_path2 = os.path.join(self.wav_dir,rel_path)
            path = os.path.join(self.root_dir, dataset_name, rel_path2)
            audio = load_audio(path)
          
            
            audio = self.audio_transform(audio)
                      
            return {"input": audio, "target": token_id}
        elif self.modality == "audiovisual":
            rel_path1 = os.path.join(self.mouth_dir,rel_path)
            path1 = os.path.join(self.root_dir, dataset_name, rel_path1)

            rel_path2 = os.path.join(self.wav_dir,rel_path)
            path2 = os.path.join(self.root_dir, dataset_name, rel_path2)

            video = load_video(path1)
            audio = load_audio(path2)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}

    def __len__(self):
        return len(self.list)
