# 1. Define the environment name and the required Python version
$VENV_NAME = "tox_env"
$PY_VERSION = "3.12"

Write-Host "--- Starting Toxicology Environment Setup ---" -ForegroundColor Cyan

# 2. Check if Python 3.12 is installed
$pyExe = py -$PY_VERSION -c "import sys; print(sys.executable)" 2>$null
if ($lastExitCode -ne 0) {
    Write-Error "Python $PY_VERSION was not found. Please install it from python.org first."
    exit
}

# 3. Create the Virtual Environment if it doesn't exist
if (-not (Test-Path -Path ".\$VENV_NAME")) {
    Write-Host "Creating Virtual Environment..." -ForegroundColor Yellow
    py -$PY_VERSION -m venv $VENV_NAME
}

# ... (previous parts of your setup script) ...

# 4. Activate and Install Dependencies
Write-Host "Installing/Updating packages..." -ForegroundColor Yellow
& ".\$VENV_NAME\Scripts\python.exe" -m pip install --upgrade pip
& ".\$VENV_NAME\Scripts\python.exe" -m pip install torch tensorboard pandas rdkit scikit-learn matplotlib setuptools

Write-Host "--- Setup Complete! ---" -ForegroundColor Green

# 5. Run the Model Script
Write-Host "Launching your PyTorch model..." -ForegroundColor Cyan
& ".\$VENV_NAME\Scripts\python.exe" .\CSC580-Mod4-CT.py