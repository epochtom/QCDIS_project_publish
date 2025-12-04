# QCDIS - Quantum Computing Data Intelligence System

A comprehensive GUI application for quantum machine learning that intelligently recommends and executes quantum ML methodologies based on the user prompts. QCDIS provides an end-to-end workflow from prompt input to model training and performance analysis.

## 🌟 Features

- **Intelligent Methodology Prediction**: A neural network recommends the best quantum ML methodology based on user prompts
- **Multiple Quantum ML Approaches**
  - **Quantum PCA (QPCA)** with Logistic Regression, CNN, or XGBoost classifiers
  - **Quantum Feature Mapping (QFM)** with Linear Regression, CNN, or XGBoost
  - **Projected Quantum Kernel (PQK)** with Neural Network
  - **MATLAB SVD-PCA** with K-Nearest Neighbors
- **Image to CSV Conversion**: Convert image datasets to CSV format for processing
- **Docker Integration**: Execute training scripts in isolated Docker containers with 24-hour timeout
- **Version History Tracking**: Track and manage script versions with automatic recovery
- **Performance Visualization**: View training results and performance metrics
- **Log Management**: Comprehensive logging and log table viewer
- **LLM Integration**: Chat interface for assistance during script development

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Supported Methodologies](#supported-methodologies)
- [Docker Setup](#docker-setup)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Docker Desktop (for training script execution)
- MATLAB (optional, for MATLAB-based methodologies)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/epochtom/QCDIS_project_publish.git
cd QCDIS_project_publish
```

2. Install Python dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow xgboost pennylane tkinter
```

3. Build the Docker image (for training script execution):
```bash
cd docker_env
docker build -t universal-training:latest -f Dockerfile ..
```

## 💻 Usage

### Starting the Application

Run the main GUI application:

```bash
python main_gui.py
```

### Workflow

The application follows a 5-page sequential workflow:

1. **Prompt Predictor Page**: Enter a description of your machine learning task. The neural network will predict the best quantum ML methodology.

2. **Image to CSV Page**: Upload and convert image datasets to CSV format, or use existing CSV files.

3. **Script Display & Execution Page**: 
   - Review the generated training script
   - Edit the script if needed
   - Execute the script in a Docker container
   - Monitor execution status

4. **Performance Viewer Page**: View training results, accuracy metrics, and performance visualizations.

5. **Log Table Viewer Page**: Browse execution logs, view version history, and analyze training runs.

## 📁 Project Structure

```
QCDIS_project/
├── main_gui.py                          # Main application entry point
├── user_interface/                      # GUI components
│   ├── prompt_predictor_gui.py         # Methodology prediction interface
│   ├── image_to_csv.py                 # Image conversion and CSV handling
│   ├── show_script_and_execute.py      # Script editor and Docker execution
│   ├── performance_viewer.py           # Performance metrics visualization
│   └── log_table_viewer.py             # Log management interface
├── universal_training_script/          # Quantum ML training scripts
│   ├── Python_QPCA_Regression.py
│   ├── Python_QPCA_CNN.py
│   ├── Python_QPCA_XGBoost.py
│   ├── Python_QFM_Regression.py
│   ├── Python_QFM_CNN.py
│   ├── Python_QFM_XGBoost.py
│   ├── Python_PQK_NN.py
│   └── Matlab_SVDPCA_KNN.m
├── neural_network_model/               # Methodology prediction model
│   ├── model.pth                       # Trained neural network (for reference only)
│   ├── vocab.pkl                       # Vocabulary for text processing
│   ├── label_encoder.pkl               # Label encoder
│   └── train_neural_network.py         # Model training script
├── docker_env/                         # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── run_script.py                   # Script execution wrapper
├── log_history_helper_functions/       # Version history management
│   ├── version_history.py
│   ├── log_parser.py
│   └── diff_calculator.py
├── LLM_integration/                    # LLM chat integration
│   └── llm_chat_widget.py
├── upload/                             # Input data directory
│   └── dataset.csv
├── output/                             # Output directory for results
└── script_versions/                    # Version history storage
```

## 🔬 Supported Methodologies

**Note**: All predefined scripts in the `universal_training_script/` directory are provided for reference only. 

### Python-Based Quantum ML

1. **Quantum PCA (QPCA)**
   - `Python_QPCA_Regression.py`: Quantum PCA with Logistic Regression
   - `Python_QPCA_CNN.py`: Quantum PCA with 1D Convolutional Neural Network
   - `Python_QPCA_XGBoost.py`: Quantum PCA with XGBoost Classifier

2. **Quantum Feature Mapping (QFM)**
   - `Python_QFM_Regression.py`: Quantum Feature Mapping with Regression
   - `Python_QFM_CNN.py`: Quantum Feature Mapping with CNN
   - `Python_QFM_XGBoost.py`: Quantum Feature Mapping with XGBoost

3. **Projected Quantum Kernel (PQK)**
   - `Python_PQK_NN.py`: Projected Quantum Kernel with Neural Network

### MATLAB-Based

4. **SVD-PCA**
   - `Matlab_SVDPCA_KNN.m`: Singular Value Decomposition PCA with K-Nearest Neighbors

## 🐳 Docker Setup

The Docker environment provides isolated execution for training scripts with a 24-hour maximum execution time.

### Building the Image

```bash
cd docker_env
docker build -t universal-training:latest -f Dockerfile ..
```

### Running Scripts

#### Using Python Wrapper (Recommended)
```bash
cd docker_env
python run_script.py Python_QPCA_Regression.py
```

#### Using PowerShell (Windows)
```powershell
cd docker_env
.\run_script.ps1 Python_QPCA_Regression.py
```

#### Using Bash (Linux/Mac)
```bash
cd docker_env
chmod +x run_script.sh
./run_script.sh Python_QPCA_Regression.py
```

### Docker Volumes

- `../upload` → `/app/data` (read-only): Input CSV files
- `../output` → `/app/output` (read-write): Output files and results
- `../universal_training_script` → `/app/universal_training_script` (read-only): Training scripts

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# OpenRouter API Key (for LLM integration)
OPENROUTER_API_KEY=your_api_key_here

# MATLAB License (for MATLAB scripts)
MATLAB_LICENSE=your_license_number
```

### Dataset Requirements

- CSV format with headers
- Target column will be auto-detected (last column or specified)
- Place datasets in the `upload/` directory

### Model Configuration

**Note**: The neural network model (`neural_network_model/model.pth`) is provided for reference only. Users may need to retrain the model with their own data for optimal performance.

Training scripts use configuration classes (CFG) that can be modified:

- `N_QUBITS`: Number of qubits for quantum circuits
- `N_COMPONENTS`: Dimensionality for dimensionality reduction
- `TEST_SIZE`: Test set split ratio
- `BATCH_SIZE`: Batch size for neural networks
- `EPOCHS`: Number of training epochs
- `FORCE_CLASSICAL`: Fallback to classical methods if True

## 🔧 Dependencies

### Python Packages
- `torch` (2.0.1): PyTorch for neural networks
- `pennylane` (0.31.1): Quantum computing framework
- `numpy` (1.23.5): Numerical computing
- `pandas` (2.0.3): Data manipulation
- `scikit-learn` (1.3.0): Machine learning utilities
- `xgboost` (2.0.0): Gradient boosting
- `matplotlib` (3.7.2): Plotting
- `pillow` (10.0.0): Image processing
- `tkinter`: GUI framework (usually included with Python)

## 📊 Output

Training scripts generate:
- Performance metrics (accuracy, F1-score)
- Visualization plots saved to `output/performance_summary.png`
- Execution logs stored in version history
- Model artifacts (if applicable)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

See [license.txt](license.txt) for details.

## 🐛 Troubleshooting

### Import Errors
- Ensure all dependencies are installed (see Installation section)
- Check that all modules are in their respective directories

### Docker Issues
- Verify Docker Desktop is running
- Check file sharing permissions in Docker settings
- Ensure the Docker image is built: `docker build -t universal-training:latest`

### MATLAB Scripts Not Running
- Verify MATLAB is installed and licensed
- Check MATLAB path configuration
- Ensure MATLAB license is set in environment variables

### Performance Issues
- Reduce the number of qubits "N_QUBITS"
- Set `FORCE_CLASSICAL=True` to use classical fallbacks
- Reduce dataset size or batch size



