# DPS-DM-lagr

Diffusion posterior sampling for the stochastic reconstruction of Lagrangian turbulent velocity signals from coarse-scale observations.

For more details, please refer to the paper *Stochastic Multiscale Reconstruction of Lagrangian Turbulence via Guided Diffusion Models*.

This repository builds on [SmartTURB/diffusion-lagr](https://github.com/SmartTURB/diffusion-lagr) and adapts unconditional diffusion models to reconstruct three-component velocity trajectories using [diffusion posterior sampling (DPS)](https://arxiv.org/abs/2209.14687). The observations are represented by Haar approximation coefficients. During reverse diffusion, a measurement-consistency gradient guides the generated trajectories toward the prescribed coarse-scale signal without retraining the diffusion model.

The main components are:

- **[continuous_diffusion](./continuous_diffusion)**: Diffusion models conditioned on a continuous noise level, following the approach described in [WaveGrad](https://arxiv.org/abs/2009.00713).
- **[tfg_diffusion](./tfg_diffusion)**: DPS guidance and task-specific observation operators, including Haar-based scale reconstruction.
- **[guided_diffusion](./guided_diffusion)**: The U-Net backbone, training utilities, and [wavelet utilities](./guided_diffusion/Wavelet.py).
- **[scripts](./scripts)**: Entry points for training, reconstruction, and recording the complete reverse-diffusion history.

The examples below use trajectories of length **1024** with **3 velocity components**.

## Installation

The environment recipe in [env_setup.txt](./env_setup.txt) uses Python 3.7 and PyTorch 1.13.1 with CUDA 11.7. The commands below retain these versions and use a consistent Conda environment name. Run them on a Linux machine with a compatible NVIDIA driver and an MPI installation available to `mpi4py` and `mpiexec`.

```bash
git clone https://github.com/QinminZheng/DPS-DM-lagr.git
cd DPS-DM-lagr

conda create -n TFG-DM-lagr python=3.7 mpi4py -y
conda activate TFG-DM-lagr

python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install numpy matplotlib
python -m pip install -e .
```

The editable installation adds the dependencies listed in [setup.py](./setup.py), including `blobfile`, `tqdm`, and `h5py`. `matplotlib` is also needed because the wavelet module imports it. The current data loaders use serial HDF5 reads, so an MPI-enabled build of `h5py` is not required. Alternative PyTorch builds are documented in the [official installation archive](https://pytorch.org/get-started/previous-versions/#v1131).

Unless specified otherwise, run the following commands from the repository root.

## Data Preparation

### Dataset: 3D HIT tracers

The source data are Lagrangian velocity trajectories of tracers in three-dimensional homogeneous isotropic turbulence (HIT). Refer to the upstream repository's [Preparing Data](https://github.com/SmartTURB/diffusion-lagr#preparing-data) section for access to `Lagr_u3c_diffusion.h5` and its normalization metadata.

For the three-component configuration used here, prepare an HDF5 file with the following structure:

| Key | Contents |
| --- | --- |
| `train` | Training trajectories with shape `(N_train, 1024, 3)` |
| `valid` | Validation trajectories with shape `(N_valid, 1024, 3)` |
| `test` | Test trajectories with shape `(N_test, 1024, 3)` |
| `min`, `max` | Min-max normalization parameters, typically one value per component |

The data loaders convert each trajectory from `(length, channels)` to `(channels, length)`. Keep the channel dimension in the stored arrays. The scripts read the selected split directly; normalization must be applied when preparing the data, using the same convention as the model checkpoint.

### Example preparation workflow

Place the original `Lagr_u3c_diffusion.h5` in `datasets/`, then run the supplied split scripts:

```bash
cd datasets
python preprocessing-lagr_u3c-diffusion-splits.py
python preprocessing-lagr_u3c-diffusion-splits-tfg.py
cd ..
```

The first script reserves the last 32,768 trajectories for testing and normalizes the data using the remaining training partition. The second shuffles that training partition with seed 0 and reserves 32,768 trajectories for validation. It writes `datasets/Lagr_u3c_diffusion_splits_tfg.h5`.

For 1024-point reconstruction, extract the first and last 1024 points from each trajectory **after splitting**, while retaining all three velocity components. The following example performs the conversion in batches and preserves the normalization metadata:

```python
import h5py

source = "datasets/Lagr_u3c_diffusion_splits_tfg.h5"
target = "datasets/Lagr_u3c_diffusion_1024_splits_tfg.h5"
length = 1024
chunk_size = 4096

with h5py.File(source, "r") as src, h5py.File(target, "x") as dst:
    for key in ("min", "max"):
        dst.create_dataset(key, data=src[key][()])

    for split in ("train", "valid", "test"):
        data = src[split]
        assert data.ndim == 3 and data.shape[2] == 3
        assert data.shape[1] >= length
        n = data.shape[0]
        out = dst.create_dataset(split, shape=(2 * n, length, 3), dtype="float32")

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            out[start:stop] = data[start:stop, :length, :]
            out[n + start:n + stop] = data[start:stop, -length:, :]
```

This doubles the number of trajectories in each split. For a 2000-point source trajectory, the two windows overlap by 48 points; for an already 1024-point trajectory, they are identical, so no window expansion is needed. Splitting before window extraction keeps windows from the same source trajectory in the same partition.

The separate script [preprocessing-lagr_u3c-split_1024-diffusion.py](./datasets/preprocessing-lagr_u3c-split_1024-diffusion.py) concatenates the three components along the sample axis and produces a **two-dimensional, single-component** array. Its output is not directly compatible with the three-channel examples above.

## Training the Unconditional Diffusion Model

DPS reconstruction uses an unconditional model trained with **`--use_continuous_diffusion True`**. The model architecture and normalization used for reconstruction must match those used during training.

The following example uses the architecture in [train.sh](./train.sh) and explicitly sets a training-step target of 400,000:

```bash
export OPENAI_LOGDIR="experiments/lagr_u3c_1024_train"
mkdir -p "$OPENAI_LOGDIR"

DATA_FLAGS="--dataset_path datasets/Lagr_u3c_diffusion_1024_splits_tfg.h5 --dataset_name train"
MODEL_FLAGS="--dims 1 --image_size 1024 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 128,64 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--use_continuous_diffusion True --diffusion_steps 800 --noise_schedule tanh6,1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --total_steps 400000 --ema_rate 0.9999"

NUM_GPUS=4
mpiexec -n "$NUM_GPUS" python scripts/turb_train.py \
    $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

`--batch_size` is the batch size per process. With four GPU processes, the example uses a global batch size of 256. Logs, model checkpoints, optimizer states, and exponential moving average (EMA) checkpoints are written to `OPENAI_LOGDIR`. An EMA checkpoint at step 400,000 is named `ema_0.9999_400000.pt`.

**GPU allocation:** [guided_diffusion/dist_util.py](./guided_diffusion/dist_util.py) currently sets `CUDA_VISIBLE_DEVICES` to `MPI_rank % GPUS_PER_NODE`, with `GPUS_PER_NODE = 8`. A single-process run selects GPU 0 under this mapping. For other node layouts or scheduler-managed jobs, adapt this mapping to the allocated devices; setting `CUDA_VISIBLE_DEVICES` in the shell alone will be overwritten by the helper.

## Model Weights

See the repository's [Releases page](https://github.com/QinminZheng/DPS-DM-lagr/releases) for release assets. To use a downloaded model, place a compatible checkpoint in `weights/` and set `--model_path` accordingly. Alternatively, use an EMA checkpoint from your own training run.

The examples use `weights/ema_0.9999_400000.pt`, matching the checkpoint filename referenced in [sample.sh](./sample.sh). This filename does not itself establish compatibility: check the architecture, sequence length, continuous-noise conditioning, and data normalization associated with the weights.

## Reconstruction with DPS

### Haar coarse-scale observations

The task `scale_reconstruction_level<level>` compares the Haar approximation coefficients of the predicted clean signal with those of the reference trajectory. In [tfg_diffusion/tasks.py](./tfg_diffusion/tasks.py), the observation operator groups every $q = 2^{\ell}$ consecutive points and computes

$$
A_{\ell}(x)_k = \sqrt{q}\,\frac{1}{q}\sum_{j=0}^{q-1}x_{kq+j}.
$$

DPS uses the gradient of the negative squared discrepancy between $A_{\ell}(x)$ and the reference coefficients to guide sampling. The reference is loaded as a full-resolution trajectory, but the guidance loss uses only its coarse-scale coefficients.

Use a nonnegative integer level for which the sequence length is divisible by `2**level`. Larger levels retain fewer coarse-scale observations.

### Sampling example

Use [scripts/tfg_sample_wave_sum_1024.py](./scripts/tfg_sample_wave_sum_1024.py) to reconstruct trajectories from the selected dataset split:

```bash
LEVEL=5
STRENGTH=0.01
SEED=42
DATASET="datasets/Lagr_u3c_diffusion_1024_splits_tfg.h5"
MODEL_PATH="weights/ema_0.9999_400000.pt"

export OPENAI_LOGDIR="experiments/dps_level${LEVEL}/GS${STRENGTH}/seed${SEED}"
mkdir -p "$OPENAI_LOGDIR"

DATA_FLAGS="--task scale_reconstruction_level${LEVEL} --dataset_path ${DATASET} --dataset_name valid"
MODEL_FLAGS="--dims 1 --image_size 1024 --in_channels 3 --num_channels 128 --num_res_blocks 3 --attention_resolutions 128,64 --channel_mult 1,1,2,3,4"
DIFFUSION_FLAGS="--diffusion_steps 800 --noise_schedule tanh6,1"
SAMPLE_FLAGS="--num_samples 64 --batch_size 64 --model_path ${MODEL_PATH} --seed ${SEED}"
GUIDANCE_FLAGS="--guidance_name dps --guidance_strength ${STRENGTH}"

python scripts/tfg_sample_wave_sum_1024.py \
    $DATA_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $GUIDANCE_FLAGS \
    > "$OPENAI_LOGDIR/sampling.log" 2>&1
```

The example uses the level and guidance strength in [sample.sh](./sample.sh), with a smaller sample count for an initial run. Set `--dataset_name test` to use the test split. Always specify `--task` explicitly: the script's current default task name is not recognized by the task loader.

`--guidance_strength` controls the coefficient-matching gradient. The current implementation applies a fixed multiplier without the residual-norm normalization used in the original DPS schedule. Tune the strength for the chosen level and checkpoint; the example is not an optimal setting for every task.

The loader cycles through the selected split. For one reconstruction per reference trajectory, keep `--num_samples` within the split size. Full batches are the simplest configuration; the example assumes at least 64 validation trajectories. Sampling seeds also depend on the process/GPU mapping, so retain that mapping when reproducing a run.

### Reading reconstructed samples

The sampling script saves a NumPy archive named `samples_<N>x1024x3-seed<seed>.npz`, with the seed padded to three digits. The reconstructed array is stored under `arr_0` and has shape `(N, 1024, 3)`.

```python
import h5py
import numpy as np

path = "experiments/dps_level5/GS0.01/seed42/samples_64x1024x3-seed042.npz"
with np.load(path) as archive:
    samples = archive["arr_0"]

with h5py.File("datasets/Lagr_u3c_diffusion_1024_splits_tfg.h5", "r") as h5f:
    lower = h5f["min"][()]
    upper = h5f["max"][()]

velocities = (samples + 1) * (upper - lower) / 2 + lower
```

The sampling scripts clamp the final normalized samples to `[-1, 1]`. The inverse transform above recovers the physical velocity values when `min` and `max` describe the corresponding physical-data normalization.

## Recording the Sampling History

[scripts/turb_sample_history_haar.py](./scripts/turb_sample_history_haar.py) records both the intermediate noisy states and the predicted clean signals during DPS reconstruction. After defining `DATASET`, `MODEL_PATH`, `MODEL_FLAGS`, and `DIFFUSION_FLAGS` as above, run:

```bash
export OPENAI_LOGDIR="experiments/dps_history/level7/GS0.16/seed42"
mkdir -p "$OPENAI_LOGDIR"

python scripts/turb_sample_history_haar.py \
    --task scale_reconstruction_level7 \
    --dataset_path "$DATASET" --dataset_name valid \
    $MODEL_FLAGS $DIFFUSION_FLAGS \
    --num_samples 4 --batch_size 4 \
    --model_path "$MODEL_PATH" --seed 42 \
    --guidance_name dps --guidance_strength 0.16 \
    > "$OPENAI_LOGDIR/history.log" 2>&1
```

With 800 diffusion steps, the output directory contains:

```text
samples_history-seed000042/
├── batch000-samples_history_4x801x1024x3.npz
└── batch000-pred_xstarts_history_4x800x1024x3.npz
```

- **`samples_history`** has shape `(batch, 801, 1024, 3)`: the initial noise followed by the 800 reverse-diffusion states.
- **`pred_xstarts_history`** has shape `(batch, 800, 1024, 3)`: the clean-signal prediction at each reverse step, ordered from the highest noise level to the lowest.

Each archive stores its array under `arr_0`; the final reconstructed trajectory is `samples_history[:, -1]`. History recording retains all intermediate states on the GPU before saving, so start with a small batch. It writes complete batches without trimming the final batch to `--num_samples`; use a sample count divisible by the effective batch size when an exact count is required.

## Notes on the Current Code

- The main workflow documented here is three-component Haar/DPS reconstruction. The `--guidance_name tfg` option is present in the argument parser, but its `guide_step` implementation is currently a placeholder. Use `dps` for the reconstruction examples.
- Legacy mask-based scripts remain in the repository, but the current palette data loader supplies only `cond` and does not accept `mask_mode`. In addition, `scripts/tfg_sample.py` imports an unavailable `tfg_diffusion_Li` module. Those entry points require updates before use.
- [train.sh](./train.sh), [sample.sh](./sample.sh), and the standalone 1024-point preprocessing script contain machine-specific paths. Use the repository-relative examples above or update the paths before running those files.
