import os
import torch
import torchaudio
import numpy as np
import torch.nn.functional as fn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# Import spikify encoders
from spikify.encoders.temporal.contrast import moving_window, step_forward
from spikify.encoders.rate import poisson


class ClefUnifiedDL:
    def __init__(self, data_dir,
        encoding_type    = "simple", # Options: "simple", "mw", "poisson", "sf"
        fft_window       = 25e-3,    # s
        hop_length_s     = 15e-3,    # s
        n_channels       = 64,       # mel bands
        spiking_thresh   = 0.2,      # used for simple, mw, and sf
        window_length    = 10,       # used for mw
        poisson_interval = 4,        # used for poisson
        train_size       = 0.8,
        random_state     = 42,       # for reproducibility
        transform        = "default",
        bipolar          = False    # whether to use bipolar encoding 
    ):
        self.sample_rate      = 32e3 # 32kHz as per recordings
        self.encoding_type    = encoding_type.lower()
        self.random_state     = random_state
        self.num_cpu_cores    = os.cpu_count()

        # Validate encoding type
        valid_encodings = ["simple", "mw", "poisson", "sf"]
        if self.encoding_type not in valid_encodings:
            raise ValueError(f"encoding_type must be one of {valid_encodings}")

        if transform == "default":
            self.transform = UnifiedSpikeTransform(
                encoding_type    = self.encoding_type,
                sample_rate      = self.sample_rate,
                fft_window       = fft_window,
                hop_length_s     = hop_length_s,
                n_mels           = n_channels,
                spiking_thresh   = spiking_thresh,
                window_length    = window_length,
                poisson_interval = poisson_interval,
                bipolar          = bipolar
            )
        else:
            self.transform = transform

        self.train_set = BCDataset(
            root_dir  = os.path.join(data_dir, "train"),
            transform = self.transform
        )
        
        self.test_set = BCDataset(
            root_dir  = os.path.join(data_dir, "test"),
            transform = self.transform
        )

        # Report split to user
        n_species = len(self.train_set.label_map)
        print(f"Loaded '{self.encoding_type}' encoding pipeline.")
        print(f"Loaded {len(self.train_set)} train / {len(self.test_set)} test samples "
              f"across {n_species} species.")

    def load(self, batch_size=16, train_shuffle=True, test_shuffle=False,
             drop_last=True, num_workers=None):

        if num_workers is None:
            num_workers = self.num_cpu_cores

        train_loader = DataLoader(self.train_set,
            batch_size  = batch_size,
            shuffle     = train_shuffle,
            num_workers = num_workers,
            drop_last   = drop_last
        )

        test_loader = DataLoader(self.test_set,
            batch_size  = batch_size,
            shuffle     = test_shuffle,
            num_workers = num_workers,
            drop_last   = drop_last
        )

        return train_loader, test_loader


class BCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir   = root_dir
        self.transform  = transform
        self.data       = []
        self.max_length = 160_000 # 5s @ 32kHz

        species = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.label_map = {name: idx for idx, name in enumerate(species)}
        print(f"Found {len(species)} species: {self.label_map}")

        for species_name, label in self.label_map.items():
            species_path = os.path.join(root_dir, species_name)
            for file_name in sorted(os.listdir(species_path)):
                if file_name.endswith(".ogg"):
                    self.data.append((os.path.join(species_path, file_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path, label = self.data[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != 32000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 32000)

        # Pad or truncate to exactly 5s @ 32kHz
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.size(1) < self.max_length:
            pad_size = self.max_length - waveform.size(1)
            waveform = fn.pad(waveform, (0, pad_size))

        if self.transform:
            waveform = self.transform(waveform)

        # Output: [T, n_mels] (or [T, n_mels * 2] if using 'sf' or 'mw') — time-first for SNN compatibility
        return waveform, label


class UnifiedSpikeTransform:
    def __init__(self, encoding_type="simple", sample_rate=32e3, fft_window=25e-3,
                 hop_length_s=15e-3, n_mels=64, spiking_thresh=0.2, window_length=10, 
                 poisson_interval=4, bipolar=False):
        
        self.encoding_type    = encoding_type
        self.sample_rate      = int(sample_rate)
        self.n_fft            = int(fft_window * sample_rate)
        self.hop_length       = int(hop_length_s * sample_rate)
        self.n_mels           = n_mels
        self.bipolar          = bipolar
        
        # Encoding-specific hyperparams
        self.spiking_thresh   = spiking_thresh
        self.window_length    = window_length
        self.poisson_interval = poisson_interval

        # Poisson encoding explicitly used center=False in the original implementation
        center_padding = False if self.encoding_type == "poisson" else True

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft       = self.n_fft,
            hop_length  = self.hop_length,
            n_mels      = self.n_mels,
            center      = center_padding
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)

        # --- POISSON ENCODING ---
        if self.encoding_type == "poisson":
            mel_spec = torch.sqrt(mel_spec)
            mel_min, mel_max = mel_spec.min(), mel_spec.max()
            mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + 1e-8)
            mel_np = mel_spec.squeeze(0).permute(1, 0).numpy()
            
            spikes = poisson(mel_np, interval_length=self.poisson_interval)
            return torch.tensor(spikes, dtype=torch.float32)

        # --- ALL OTHER ENCODINGS (Require dB and standard normalization) ---
        mel_spec = self.db_transform(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        if self.encoding_type == "simple":
            mel_spec = (mel_spec > self.spiking_thresh).float()
            # Reshape [1, n_mels, T] -> [T, n_mels] for SNN compatibility
            return mel_spec.squeeze(0).permute(1, 0)
            
        # Convert to numpy for spikify library tools
        mel_np = mel_spec.squeeze(0).permute(1, 0).numpy()

        if self.encoding_type == "mw" or self.encoding_type == "sf":
        
            if self.encoding_type == "mw":
                raw_spikes, _ = moving_window(mel_np, self.window_length, self.spiking_thresh)

            else: # step forward
                raw_spikes, _ = step_forward(mel_np, threshold=self.spiking_thresh)
            
            spikes_tensor = torch.tensor(raw_spikes, dtype=torch.float32)
                
            pos_spikes = (spikes_tensor == 1.0).float()
            neg_spikes = (spikes_tensor == -1.0).float()
            
            if self.bipolar:
                return torch.tensor(raw_spikes, dtype=torch.float32)
                
            else:
                # Concatenate along the final dimension (n_mels)
                # Resulting shape turns from [T, n_mels] into [T, n_mels * 2]
                return torch.cat((pos_spikes, neg_spikes), dim=-1)
                

        raise ValueError("Invalid encoding type processed.")