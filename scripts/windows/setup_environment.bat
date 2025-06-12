@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM Script di Setup Automatico Environment ViT Fine-tuning (Windows)
REM =============================================================================

echo 🚀 Setup Automatico Environment ViT Fine-tuning
echo =================================================
echo Questo script creerà l'environment 'vit-training' con tutte le dipendenze necessarie
echo.

REM Controlla se conda è disponibile
where conda >nul 2>&1
if errorlevel 1 (
    echo ❌ ERRORE: Conda non trovato nel PATH
    echo    Assicurati di aver installato Anaconda/Miniconda e di averlo aggiunto al PATH
    echo    Oppure esegui questo script dall'Anaconda Prompt
    pause
    exit /b 1
)

echo ✅ Conda trovato: 
conda --version

REM Controlla se l'environment esiste già
echo.
echo 🔍 Controllo se l'environment 'vit-training' esiste già...
conda env list | findstr "vit-training" >nul 2>&1
if not errorlevel 1 (
    echo ⚠️  L'environment 'vit-training' esiste già!
    set /p choice="Vuoi rimuoverlo e ricrearlo? (y/n): "
    if /i "!choice!"=="y" (
        echo 🗑️  Rimozione environment esistente...
        conda env remove -n vit-training -y
        if errorlevel 1 (
            echo ❌ Errore nella rimozione dell'environment
            pause
            exit /b 1
        )
        echo ✅ Environment rimosso
    ) else (
        echo ⏭️  Installazione delle dipendenze nell'environment esistente...
        goto :install_packages
    )
)

REM Crea nuovo environment
echo.
echo 📦 Creazione environment 'vit-training' con Python 3.10...
conda create -n vit-training python=3.10 -y
if errorlevel 1 (
    echo ❌ Errore nella creazione dell'environment
    pause
    exit /b 1
)
echo ✅ Environment creato con successo

:install_packages
echo.
echo 🔧 Installazione dipendenze...

REM Attiva environment (per il resto dello script)
call conda activate vit-training
if errorlevel 1 (
    echo ❌ Errore nell'attivazione dell'environment
    pause
    exit /b 1
)

echo ✅ Environment 'vit-training' attivato

REM Installa PyTorch e dipendenze conda
echo.
echo 🧠 Installazione PyTorch e dipendenze conda...
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
if errorlevel 1 (
    echo ❌ Errore nell'installazione di PyTorch
    pause
    exit /b 1
)
echo ✅ PyTorch installato

echo.
echo 📊 Installazione librerie scientifiche...
conda install -c conda-forge numpy==1.24.3 pandas==2.0.1 scipy==1.10.0 -y
if errorlevel 1 (
    echo ❌ Errore nell'installazione delle librerie scientifiche
    pause
    exit /b 1
)
echo ✅ Librerie scientifiche installate

REM Installa dipendenze pip
echo.
echo 📦 Installazione dipendenze Python via pip...

echo   - jsonargparse...
pip install jsonargparse==4.21.1
if errorlevel 1 (
    echo ❌ Errore nell'installazione di jsonargparse
    pause
    exit /b 1
)

echo   - peft...
pip install peft==0.3.0
if errorlevel 1 (
    echo ❌ Errore nell'installazione di peft
    pause
    exit /b 1
)

echo   - pytorch_lightning...
pip install pytorch_lightning[extra]==2.0.2
if errorlevel 1 (
    echo ❌ Errore nell'installazione di pytorch_lightning
    pause
    exit /b 1
)

echo   - torchmetrics...
pip install torchmetrics==0.11.4
if errorlevel 1 (
    echo ❌ Errore nell'installazione di torchmetrics
    pause
    exit /b 1
)

echo   - transformers...
pip install transformers==4.38.0
if errorlevel 1 (
    echo ❌ Errore nell'installazione di transformers
    pause
    exit /b 1
)

echo   - wandb...
pip install wandb==0.15.2
if errorlevel 1 (
    echo ❌ Errore nell'installazione di wandb
    pause
    exit /b 1
)

echo   - tensorboard...
pip install tensorboard
if errorlevel 1 (
    echo ❌ Errore nell'installazione di tensorboard
    pause
    exit /b 1
)

echo ✅ Tutte le dipendenze pip installate

REM Verifica installazione
echo.
echo 🔍 Verifica installazione...
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import pytorch_lightning; print('✅ Lightning:', pytorch_lightning.__version__)"
python -c "import transformers; print('✅ Transformers:', transformers.__version__)"
python -c "import tensorboard; print('✅ TensorBoard:', tensorboard.__version__)"

echo.
echo 🎉 SETUP COMPLETATO CON SUCCESSO!
echo =================================================
echo Environment: vit-training
echo Python: 3.10
echo PyTorch: 2.2.0 (CUDA 11.8)
echo Lightning: 2.0.2
echo =================================================
echo.
echo 📋 PROSSIMI PASSI:
echo 1. Attiva l'environment: conda activate vit-training
echo 2. Vai al progetto: cd ..\..\
echo 3. Esegui esperimenti: scripts\windows\run_experiments.bat
echo.
echo 💡 TIP: Aggiungi il dataset in data\custom_dataset\
echo.

pause 