@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM Script per eseguire un singolo esperimento ViT Fine-tuning (Windows Version)
REM =============================================================================

REM Parametri di default (modificabili via argomenti)
set DATASET_ROOT=%1
set NUM_CLASSES=%2
set TRAINING_MODE=%3
set OPTIMIZER=%4
set LR=%5
set WEIGHT_DECAY=%6
set WARMUP_STEPS=%7
set BATCH_SIZE=%8
set LORA_R=%9

REM Imposta valori di default se non forniti
if "%DATASET_ROOT%"=="" set DATASET_ROOT=..\..\data\custom_dataset
if "%NUM_CLASSES%"=="" set NUM_CLASSES=10
if "%TRAINING_MODE%"=="" set TRAINING_MODE=full
if "%OPTIMIZER%"=="" set OPTIMIZER=sgd
if "%LR%"=="" set LR=0.01
if "%WEIGHT_DECAY%"=="" set WEIGHT_DECAY=0.0
if "%WARMUP_STEPS%"=="" set WARMUP_STEPS=500
if "%BATCH_SIZE%"=="" set BATCH_SIZE=128
if "%LORA_R%"=="" set LORA_R=8

REM LORA_ALPHA come 10° parametro (shift per accederlo)
shift
set LORA_ALPHA=%9
if "%LORA_ALPHA%"=="" set LORA_ALPHA=16

REM Configurazioni fisse
set IMAGE_SIZE=224
set MAX_STEPS=5000
set VAL_CHECK_INTERVAL=500

REM Genera nome esperimento
set EXP_NAME=%TRAINING_MODE%_%OPTIMIZER%_lr%LR:~0,3%%LR:~4%_wd%WEIGHT_DECAY:~0,1%%WEIGHT_DECAY:~2%_w%WARMUP_STEPS%_b%BATCH_SIZE%

if "%TRAINING_MODE%"=="lora" (
    set EXP_NAME=%EXP_NAME%_r%LORA_R%_a%LORA_ALPHA%
)

echo 🚀 Avvio esperimento singolo: %EXP_NAME%
echo =================================================
echo Dataset: %DATASET_ROOT%
echo Classi: %NUM_CLASSES%
echo Training Mode: %TRAINING_MODE%
echo Optimizer: %OPTIMIZER%
echo Learning Rate: %LR%
echo Weight Decay: %WEIGHT_DECAY%
echo Warmup Steps: %WARMUP_STEPS%
echo Batch Size: %BATCH_SIZE%

if "%TRAINING_MODE%"=="lora" (
    echo LoRA r: %LORA_R%
    echo LoRA alpha: %LORA_ALPHA%
)

echo =================================================

REM Crea directory output
if not exist "..\..\output\single_experiments" mkdir "..\..\output\single_experiments"

REM Esegui esperimento
echo.
echo Eseguendo esperimento...
echo.

if "%TRAINING_MODE%"=="lora" (
    python ..\..\main.py fit ^
        --trainer.accelerator gpu ^
        --trainer.devices 1 ^
        --trainer.precision 16-mixed ^
        --trainer.max_steps %MAX_STEPS% ^
        --trainer.val_check_interval %VAL_CHECK_INTERVAL% ^
        --trainer.logger.class_path pytorch_lightning.loggers.CSVLogger ^
        --trainer.logger.init_args.save_dir ..\..\output\single_experiments ^
        --trainer.logger.init_args.name "%EXP_NAME%" ^
        --model.model_name vit-b16-224-in21k ^
        --model.training_mode %TRAINING_MODE% ^
        --model.optimizer %OPTIMIZER% ^
        --model.lr %LR% ^
        --model.weight_decay %WEIGHT_DECAY% ^
        --model.warmup_steps %WARMUP_STEPS% ^
        --model.scheduler cosine ^
        --model.lora_r %LORA_R% ^
        --model.lora_alpha %LORA_ALPHA% ^
        --model.lora_target_modules "['query', 'value']" ^
        --data.dataset custom ^
        --data.root "%DATASET_ROOT%" ^
        --data.num_classes %NUM_CLASSES% ^
        --data.size %IMAGE_SIZE% ^
        --data.batch_size %BATCH_SIZE% ^
        --data.workers 4 ^
        --model_checkpoint.filename "best-step-{step}-{val_acc:.4f}" ^
        --model_checkpoint.monitor val_acc ^
        --model_checkpoint.mode max ^
        --model_checkpoint.save_last true
) else (
    python ..\..\main.py fit ^
        --trainer.accelerator gpu ^
        --trainer.devices 1 ^
        --trainer.precision 16-mixed ^
        --trainer.max_steps %MAX_STEPS% ^
        --trainer.val_check_interval %VAL_CHECK_INTERVAL% ^
        --trainer.logger.class_path pytorch_lightning.loggers.CSVLogger ^
        --trainer.logger.init_args.save_dir ..\..\output\single_experiments ^
        --trainer.logger.init_args.name "%EXP_NAME%" ^
        --model.model_name vit-b16-224-in21k ^
        --model.training_mode %TRAINING_MODE% ^
        --model.optimizer %OPTIMIZER% ^
        --model.lr %LR% ^
        --model.weight_decay %WEIGHT_DECAY% ^
        --model.warmup_steps %WARMUP_STEPS% ^
        --model.scheduler cosine ^
        --data.dataset custom ^
        --data.root "%DATASET_ROOT%" ^
        --data.num_classes %NUM_CLASSES% ^
        --data.size %IMAGE_SIZE% ^
        --data.batch_size %BATCH_SIZE% ^
        --data.workers 4 ^
        --model_checkpoint.filename "best-step-{step}-{val_acc:.4f}" ^
        --model_checkpoint.monitor val_acc ^
        --model_checkpoint.mode max ^
        --model_checkpoint.save_last true
)

echo.
echo ✅ Esperimento completato: %EXP_NAME%
echo 📁 Risultati salvati in: ..\..\output\single_experiments\%EXP_NAME%

REM Chiedi se testare il modello
echo.
set /p reply="Vuoi testare il modello appena addestrato? (y/n): "
if /i "%reply%"=="y" (
    echo 🔍 Avvio test del modello...
    
    REM Trova l'ultimo checkpoint
    for /d %%i in (..\..\output\single_experiments\%EXP_NAME%\version_*) do (
        set LATEST_VERSION=%%i
    )
    
    if exist "!LATEST_VERSION!\checkpoints" if exist "!LATEST_VERSION!\config.yaml" (
        for %%j in ("!LATEST_VERSION!\checkpoints\best-*.ckpt") do (
            if exist "%%j" (
                python ..\..\main.py test ^
                    --ckpt_path "%%j" ^
                    --config "!LATEST_VERSION!\config.yaml" ^
                    --trainer.precision 16-mixed
                
                echo ✅ Test completato!
                goto :test_done
            )
        )
        echo ❌ Checkpoint non trovato
    ) else (
        echo ❌ File di configurazione non trovato
    )
)

:test_done
echo.
echo 📊 Per analizzare tutti i risultati:
echo    python ..\..\analyze_results.py --experiments_dir ..\..\output\single_experiments

REM =============================================================================
REM HELP
REM =============================================================================

if "%1"=="--help" (
    echo.
    echo SINTASSI:
    echo    single_experiment.bat [dataset] [classi] [mode] [optimizer] [lr] [wd] [warmup] [batch] [lora_r] [lora_alpha]
    echo.
    echo ESEMPI:
    echo    REM Full fine-tuning
    echo    single_experiment.bat ..\..\data\custom_dataset 10 full sgd 0.01 0.0 500 128
    echo.
    echo    REM LoRA fine-tuning
    echo    single_experiment.bat ..\..\data\custom_dataset 10 lora sgd 0.05 0.0 500 128 8 16
    echo.
    echo    REM Linear probe
    echo    single_experiment.bat ..\..\data\custom_dataset 10 linear sgd 0.1 0.0 100 128
    echo.
) 