# Docker Environment for Universal Training Scripts

This Docker environment allows you to execute any script from the `universal_training_script` folder in an isolated container with a 24-hour maximum execution time.

## Prerequisites

- Docker installed and running
- Docker Compose (optional, for docker-compose.yml)

## Quick Start

### Using Python Script (Recommended for Windows)

```bash
cd docker_env
python run_script.py QPCA_Regression.py
```

### Using PowerShell Script (Windows)

```powershell
cd docker_env
.\run_script.ps1 QPCA_Regression.py
```

### Using Bash Script (Linux/Mac)

```bash
cd docker_env
chmod +x run_script.sh
./run_script.sh QPCA_Regression.py
```

### Using Docker Compose

```bash
cd docker_env
docker-compose run --rm training-script python universal_training_script/QPCA_Regression.py
```

## Available Scripts

- `QPCA_Regression.py` - Quantum PCA with Regression
- `QPCA_CNN.py` - Quantum PCA with CNN
- `QFM_Regression.py` - Quantum Feature Mapping with Regression
- `QFM_CNN.py` - Quantum Feature Mapping with CNN

## Usage Examples

### Basic execution:
```bash
python run_script.py QPCA_Regression.py
```

### With additional arguments:
```bash
python run_script.py QPCA_CNN.py --arg1 value1 --arg2 value2
```

### Using Docker directly:
```bash
docker run --rm \
  --stop-timeout 86400 \
  -v "$(pwd)/../upload:/app/data:ro" \
  -v "$(pwd)/../output:/app/output:rw" \
  -v "$(pwd)/../universal_training_script:/app/universal_training_script:ro" \
  -w /app \
  universal-training:latest \
  python universal_training_script/QPCA_Regression.py
```

## Directory Structure

```
platform/
├── docker_env/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── run_script.py
│   ├── run_script.ps1
│   ├── run_script.sh
│   └── README.md
├── universal_training_script/
│   ├── QPCA_Regression.py
│   ├── QPCA_CNN.py
│   ├── QFM_Regression.py
│   └── QFM_CNN.py
├── upload/          # Input data directory (mounted as /app/data)
└── output/          # Output directory (mounted as /app/output)
```

## Configuration

### Timeout
The Docker container has a maximum execution time of **24 hours (86400 seconds)**. This is set via:
- `--stop-timeout 86400` in docker run commands
- `stop_grace_period: 86400s` in docker-compose.yml

### Volumes
- `../upload` → `/app/data` (read-only) - Input CSV files
- `../output` → `/app/output` (read-write) - Output files and results
- `../universal_training_script` → `/app/universal_training_script` (read-only) - Scripts to execute

### Environment Variables
- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered
- `PYTHONPATH=/app` - Sets Python path

## Dependencies

The Docker image includes:
- Python 3.10
- NumPy 1.24.3
- Pandas 2.0.3
- Scikit-learn 1.3.0
- PyTorch 2.0.1
- PennyLane 0.31.1
- Matplotlib 3.7.2
- Pillow 10.0.0

## Building the Image

To build the Docker image manually:

```bash
cd docker_env
docker build -t universal-training:latest -f Dockerfile ..
```

## Troubleshooting

### Container times out before 24 hours
- Check Docker daemon timeout settings
- Verify system resources (memory, CPU)
- Check if the script is actually running or stuck

### Permission errors
- Ensure Docker has access to the mounted directories
- On Windows, check Docker Desktop file sharing settings

### Script not found
- Verify the script name is correct
- Check that the script exists in `universal_training_script` folder

### Build fails
- Ensure Docker is running
- Check internet connection (for downloading packages)
- Verify Dockerfile syntax

## Notes

- The container runs in isolated mode with no network access (unless needed)
- All input files should be placed in the `upload` folder
- All output files will be saved to the `output` folder
- Scripts can access data from `/app/data` and write to `/app/output`

