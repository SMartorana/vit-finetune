@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM Script per preparare la struttura del dataset custom (Windows Version)
REM =============================================================================

echo 🚀 Setup Dataset Custom per ViT Fine-tuning
echo ============================================

REM Configurazione
set DATASET_DIR=..\..\data\custom_dataset
set NUM_CLASSES=10

REM Crea struttura directory
echo 📁 Creazione struttura directory...
if not exist "%DATASET_DIR%" mkdir "%DATASET_DIR%"

REM Crea directory per train/val/test splits
for %%s in (train val test) do (
    if not exist "%DATASET_DIR%\%%s" mkdir "%DATASET_DIR%\%%s"
    
    REM Crea directory per ogni classe (esempio con 10 classi)
    for /L %%i in (0,1,9) do (
        if %%i LSS 10 (
            set class_name=class_0%%i
        ) else (
            set class_name=class_%%i
        )
        
        if not exist "%DATASET_DIR%\%%s\!class_name!" mkdir "%DATASET_DIR%\%%s\!class_name!"
        echo    ✅ Creata: %DATASET_DIR%\%%s\!class_name!
    )
)

echo.
echo 📊 Struttura dataset creata:
echo %DATASET_DIR%\
echo ├── train\
echo │   ├── class_00\  # ^<- Inserisci qui le immagini di training
echo │   ├── class_01\
echo │   ├── ...
echo │   └── class_09\
echo ├── val\
echo │   ├── class_00\  # ^<- Inserisci qui le immagini di validazione
echo │   ├── class_01\
echo │   ├── ...
echo │   └── class_09\
echo └── test\
echo     ├── class_00\  # ^<- Inserisci qui le immagini di test
echo     ├── class_01\
echo     ├── ...
echo     └── class_09\
echo.

echo 💡 ISTRUZIONI:
echo 1. Sostituisci 'class_XX' con i nomi reali delle tue classi
echo 2. Inserisci ~800 immagini per classe in train\
echo 3. Inserisci ~100 immagini per classe in val\
echo 4. Inserisci ~100 immagini per classe in test\
echo 5. Formati supportati: .jpg, .jpeg, .png, .bmp, .tiff
echo.

echo 🔧 CONFIGURAZIONE:
echo Negli script di training, il dataset viene automaticamente configurato come:
echo    DATASET_ROOT=%DATASET_DIR%
echo    NUM_CLASSES=%NUM_CLASSES%
echo.

echo ✅ Setup completato!
echo 🚀 Prossimi passi:
echo    1. Popola il dataset con le tue immagini
echo    2. Esegui: run_experiments.bat
echo    3. Analizza risultati: python ..\..\analyze_results.py

echo.
echo 📝 HELP:
echo Per vedere la sintassi degli script:
echo    single_experiment.bat --help

pause 