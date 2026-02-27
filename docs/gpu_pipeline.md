# GPU Wavelet Heartbeat Monitor – Architecture

OpenCL-accelerated pipeline on branch `feature/gpu-wavelet-opencl`.

```mermaid
flowchart TD
    subgraph ENTRY["main_wavelet_gpu.py  (entry point)"]
        CLI["CLI args parser\n--resolution --fps --window\n--show-bands --cpu-fallback --info"]
    end

    subgraph HW["heartbeat_monitor/gpu/hardware_detector.py"]
        CPU_INFO["/proc/cpuinfo\n/proc/device-tree/model"]
        BOARD["BoardType detection\nRPI5 · RPI_ZERO_2W · RPI_OTHER\nGENERIC_ARM · X86 · UNKNOWN"]
        CL_ENUM["OpenCL platform/device enumeration\npyopencl.get_platforms()"]
        PICK["_pick_best_device()\nGPU → ACCELERATOR → CPU"]
        TUNING["HardwareProfile\n preferred_work_group\n max_wavelet_scales\n signal_buffer_limit\n use_float16"]
        CPU_INFO --> BOARD
        CL_ENUM --> PICK
        BOARD --> TUNING
        PICK --> TUNING
    end

    subgraph KERN["heartbeat_monitor/gpu/wavelet_kernels.py  (OpenCL C source)"]
        K1["detrend_normalize\nglobal_size=(N,)\nsubtracts mean, divides by std"]
        K2["cwt_morlet\nglobal_size=(n_scales, N)\ncomplex Morlet CWT → Re²+Im² power"]
        K3["cwt_morlet_tiled\nglobal_size=(n_scales, N)  WG=(1,64)\nRPi Zero 2W path — __local tile cache"]
        K4["band_energy\nglobal_size=(n_scales,)\nsum power over time → energy per scale"]
    end

    subgraph PROC["heartbeat_monitor/gpu/wavelet_processor_gpu.py"]
        PUSH["push_frame(bgr)\nmean green channel → deque buffer"]
        DETREND_H["Host detrend\nnumpy mean/std  ~μs"]
        UP["H→D upload\nsignal float32  ~bytes × 4"]
        GPU_RUN["GPU pipeline\n1 detrend_normalize\n2 cwt_morlet  or  cwt_morlet_tiled\n3 band_energy"]
        DOWN["D→H download\nenergy[n_scales]  ~128 bytes"]
        PEAK["Host: argmax(energy)\n→ scale_to_bpm()  →  BPM, confidence"]
        FALLBACK["NumPy CPU fallback\n_numpy_cwt_energy()\nactivated if no pyopencl / no device"]
        PUSH --> DETREND_H --> UP --> GPU_RUN --> DOWN --> PEAK
        DETREND_H -.->|"no OpenCL"| FALLBACK --> PEAK
    end

    ENTRY --> HW
    HW --> PROC
    KERN --> GPU_RUN

    subgraph SHARED["Shared — unchanged from main branch"]
        CAM["IMX500Camera\ncamera.py"]
        FD["FingerDetector\nfinger_detector.py"]
        VIS["Visualizer\nvisualizer.py\n+ band-energy bar overlay"]
    end

    ENTRY --> CAM
    ENTRY --> FD
    ENTRY --> VIS

    subgraph TUNING_TABLE["Board tuning table"]
        T1["RPi 5      WG=128  scales=32  buf=1024  fp16=off"]
        T2["RPi Zero2W WG=64   scales=16  buf=512   fp16=on  → tiled kernel"]
        T3["RPi other  WG=64   scales=24  buf=512   fp16=off"]
        T4["x86        WG=256  scales=48  buf=2048  fp16=off"]
    end

    TUNING --> TUNING_TABLE
```

## Pipeline description

### Per-frame data flow

```
push_frame(bgr)  →  extract mean green channel  →  rolling deque
        │
        ▼  (every 3rd frame, once min_samples reached)
Host: numpy detrend  (subtract mean, divide std)          ~μs
        │
        ▼
H → D  upload float32 signal                              ~N × 4 bytes
        │
        ├─ GPU kernel 1: detrend_normalize   global=(N,)
        │
        ├─ GPU kernel 2: cwt_morlet          global=(n_scales, N)
        │       or       cwt_morlet_tiled    global=(n_scales, N)  WG=(1,64)
        │                                   [RPi Zero 2W local-mem path]
        │
        └─ GPU kernel 3: band_energy         global=(n_scales,)
                │
                ▼
D → H  download energy[n_scales]                          ~n_scales × 4 bytes
        │
        ▼
Host: argmax(energy) → scale_to_bpm() → BPM + confidence  ~μs
```

### OpenCL kernels

| Kernel | Global size | Purpose |
|--------|-------------|---------|
| `detrend_normalize` | `(N,)` | Parallel subtract mean, divide std-dev |
| `cwt_morlet` | `(n_scales, N)` | One work-item per `(scale, time)`: computes complex Morlet CWT power `Re² + Im²` |
| `cwt_morlet_tiled` | `(n_scales, N)` WG `(1, 64)` | RPi Zero 2W variant: loads signal tile into `__local` memory to reduce global memory pressure |
| `band_energy` | `(n_scales,)` | Sum power over all time positions for each scale → scalar energy |

### Hardware tuning

| Board | WG size | CWT scales | Buffer (samples) | fp16 | Kernel |
|-------|---------|-----------|-----------------|------|--------|
| Raspberry Pi 5 (VideoCore VII) | 128 | 32 | 1024 | off | standard |
| Raspberry Pi Zero 2W (VideoCore VI) | 64 | 16 | 512 | on | **tiled** |
| Raspberry Pi other (3B / 4B) | 64 | 24 | 512 | off | standard |
| Generic ARM | 128 | 32 | 1024 | off | standard |
| x86 desktop | 256 | 48 | 2048 | off | standard |

### Graceful degradation

If `pyopencl` is not installed or no OpenCL platform is available the processor
falls back automatically to `_numpy_cwt_energy()` — the same Morlet CWT
algorithm implemented in pure NumPy.  The public API is identical in both paths.

## Running

```bash
# Print hardware/OpenCL detection report
./run_wavelet_gpu.sh --info

# Live window with GPU acceleration
./run_wavelet_gpu.sh

# Add band-energy histogram overlay
./run_wavelet_gpu.sh --show-bands

# Force CPU-only (no OpenCL required)
./run_wavelet_gpu.sh --cpu-fallback

# Headless, log BPM to stdout
./run_wavelet_gpu.sh --headless

# Record annotated video
./run_wavelet_gpu.sh --save output.mp4
```
