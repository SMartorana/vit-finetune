@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM ViT Fine-tuning Experiment Script (Windows Version)
REM Custom Dataset: ~10 classi, ~1000 immagini per classe
REM =============================================================================

REM Configurazione dataset custom
set DATASET_ROOT=..\..\data\custom_dataset
set NUM_CLASSES=10
set IMAGE_SIZE=224
set MAX_STEPS=5000
set VAL_CHECK_INTERVAL=500

REM Crea directory output se non esiste
if not exist "..\..\output\experiments" mkdir "..\..\output\experiments"

echo 🎯 Avvio esperimenti ViT Fine-tuning
echo Dataset: %DATASET_ROOT%
echo Classi: %NUM_CLASSES%
echo Image Size: %IMAGE_SIZE%
echo =================================================

REM Avvia TensorBoard in background per monitoraggio live
echo 📊 Avvio TensorBoard per monitoraggio live...
start /B tensorboard --logdir "..\..\output\experiments" --port 6006 --host 0.0.0.0
timeout /t 3 /nobreak >nul

REM Apri TensorBoard nel browser
echo 🌐 Apertura TensorBoard nel browser...
start http://localhost:6006
echo 📈 TensorBoard disponibile su: http://localhost:6006
echo =================================================

REM =============================================================================
REM ESPERIMENTI FULL FINE-TUNING
REM =============================================================================

echo.
echo 🔥 FULL FINE-TUNING EXPERIMENTS

REM Full + SGD + Different LRs
call :run_experiment "full_sgd_lr001_wd0_w500_b128" "full" 0.01 "sgd" 0.0 500 128
call :run_experiment "full_sgd_lr003_wd0_w500_b128" "full" 0.03 "sgd" 0.0 500 128
call :run_experiment "full_sgd_lr005_wd0_w500_b128" "full" 0.05 "sgd" 0.0 500 128

REM Full + SGD + Weight Decay
call :run_experiment "full_sgd_lr01_wd1e4_w500_b128" "full" 0.01 "sgd" 1e-4 500 128
call :run_experiment "full_sgd_lr01_wd1e3_w500_b128" "full" 0.01 "sgd" 1e-3 500 128

REM Full + SGD + Different Warmup
call :run_experiment "full_sgd_lr01_wd0_w100_b128" "full" 0.01 "sgd" 0.0 100 128
call :run_experiment "full_sgd_lr01_wd0_w1000_b128" "full" 0.01 "sgd" 0.0 1000 128

REM Full + AdamW
call :run_experiment "full_adamw_lr001_wd1e4_w500_b128" "full" 0.001 "adamw" 1e-4 500 128
call :run_experiment "full_adamw_lr003_wd1e4_w500_b128" "full" 0.003 "adamw" 1e-4 500 128

REM Full + Different Batch Sizes
call :run_experiment "full_sgd_lr01_wd0_w500_b64" "full" 0.01 "sgd" 0.0 500 64
call :run_experiment "full_sgd_lr01_wd0_w500_b256" "full" 0.01 "sgd" 0.0 500 256

REM =============================================================================
REM ESPERIMENTI LoRA FINE-TUNING
REM =============================================================================

echo.
echo ⚡ LoRA FINE-TUNING EXPERIMENTS

REM LoRA + Different r values
call :run_lora_experiment "lora_r1_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 1 16
call :run_lora_experiment "lora_r4_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 4 16
call :run_lora_experiment "lora_r8_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 8 16
call :run_lora_experiment "lora_r16_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 16 16

REM LoRA + Different alpha values
call :run_lora_experiment "lora_r8_a8_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 8 8
call :run_lora_experiment "lora_r8_a32_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 8 32

REM LoRA + Different LRs
call :run_lora_experiment "lora_r8_a16_sgd_lr001_wd0_w500_b128" "lora" 0.01 "sgd" 0.0 500 128 8 16
call :run_lora_experiment "lora_r8_a16_sgd_lr003_wd0_w500_b128" "lora" 0.03 "sgd" 0.0 500 128 8 16
call :run_lora_experiment "lora_r8_a16_sgd_lr01_wd0_w500_b128" "lora" 0.1 "sgd" 0.0 500 128 8 16

REM LoRA + AdamW
call :run_lora_experiment "lora_r8_a16_adamw_lr001_wd1e4_w500_b128" "lora" 0.001 "adamw" 1e-4 500 128 8 16
call :run_lora_experiment "lora_r8_a16_adamw_lr003_wd1e4_w500_b128" "lora" 0.003 "adamw" 1e-4 500 128 8 16

REM =============================================================================
REM ESPERIMENTI LINEAR PROBE
REM =============================================================================

echo.
echo 📊 LINEAR PROBE EXPERIMENTS

REM Linear + Different LRs
call :run_experiment "linear_sgd_lr01_wd0_w100_b128" "linear" 0.1 "sgd" 0.0 100 128
call :run_experiment "linear_sgd_lr05_wd0_w100_b128" "linear" 0.5 "sgd" 0.0 100 128
call :run_experiment "linear_sgd_lr1_wd0_w100_b128" "linear" 1.0 "sgd" 0.0 100 128

REM Linear + AdamW
call :run_experiment "linear_adamw_lr001_wd1e4_w100_b128" "linear" 0.001 "adamw" 1e-4 100 128
call :run_experiment "linear_adamw_lr01_wd1e4_w100_b128" "linear" 0.01 "adamw" 1e-4 100 128

echo.
echo 🎉 Tutti gli esperimenti completati!
echo 📁 Risultati salvati in: ..\..\output\experiments\
echo =================================================

REM =============================================================================
REM TESTING DEI MIGLIORI MODELLI
REM =============================================================================

echo.
echo 🔍 AVVIO TESTING AUTOMATICO
set /p reply="Vuoi procedere con il testing automatico dei modelli? (y/n): "
if /i "%reply%" == "y" (
    echo 🚀 Avvio testing dei modelli...
    call :test_all_models
    echo ✅ Testing completato per tutti i modelli!
)

echo.
echo 📊 Per analizzare i risultati:
echo python ..\..\analyze_results.py

echo.
echo 🛑 TensorBoard è ancora in esecuzione su http://localhost:6006
echo    Per chiuderlo manualmente: taskkill /F /IM tensorboard.exe
echo    O chiudi semplicemente questa finestra del terminale

goto :eof

REM =============================================================================
REM FUNZIONI
REM =============================================================================

:run_experiment
set exp_name=%~1
set training_mode=%~2
set lr=%~3
set optimizer=%~4
set weight_decay=%~5
set warmup_steps=%~6
set batch_size=%~7

echo.
echo 🚀 Avvio esperimento: %exp_name%
echo    Training Mode: %training_mode% ^| LR: %lr% ^| Optimizer: %optimizer%
echo    Weight Decay: %weight_decay% ^| Warmup: %warmup_steps% ^| Batch: %batch_size%

python ..\..\main.py fit ^
    --trainer.accelerator gpu ^
    --trainer.devices 1 ^
    --trainer.precision 16-mixed ^
    --trainer.max_steps %MAX_STEPS% ^
    --trainer.val_check_interval %VAL_CHECK_INTERVAL% ^
    --trainer.logger.class_path pytorch_lightning.loggers.TensorBoardLogger ^
    --trainer.logger.init_args.save_dir ..\..\output\experiments ^
    --trainer.logger.init_args.name "%exp_name%" ^
    --model.model_name vit-b16-224-in21k ^
    --model.training_mode %training_mode% ^
    --model.optimizer %optimizer% ^
    --model.lr %lr% ^
    --model.weight_decay %weight_decay% ^
    --model.warmup_steps %warmup_steps% ^
    --model.scheduler cosine ^
    --data.dataset custom ^
    --data.root "%DATASET_ROOT%" ^
    --data.num_classes %NUM_CLASSES% ^
    --data.size %IMAGE_SIZE% ^
    --data.batch_size %batch_size% ^
    --data.workers 4 ^
    --model_checkpoint.filename "best-step-{step}-{val_acc:.4f}" ^
    --model_checkpoint.monitor val_acc ^
    --model_checkpoint.mode max ^
    --model_checkpoint.save_last true

REM Export TensorBoard logs to CSV
echo 📊 Esportazione TensorBoard logs in CSV...
for /d %%v in (..\..\output\experiments\%exp_name%\version_*) do (
    if exist "%%v" (
        tensorboard --logdir "%%v" --export_to_csv "%%v\metrics.csv" 2>nul || echo ⚠️  TensorBoard export fallito per %%v
    )
)

echo ✅ Completato: %exp_name%
echo ---------------------------------------------------
goto :eof

:run_lora_experiment
set exp_name=%~1
set training_mode=%~2
set lr=%~3
set optimizer=%~4
set weight_decay=%~5
set warmup_steps=%~6
set batch_size=%~7
set lora_r=%~8
set lora_alpha=%~9

echo.
echo 🚀 Avvio esperimento: %exp_name%
echo    Training Mode: %training_mode% ^| LR: %lr% ^| Optimizer: %optimizer%
echo    Weight Decay: %weight_decay% ^| Warmup: %warmup_steps% ^| Batch: %batch_size%
echo    LoRA r: %lora_r% ^| LoRA alpha: %lora_alpha%

python ..\..\main.py fit ^
    --trainer.accelerator gpu ^
    --trainer.devices 1 ^
    --trainer.precision 16-mixed ^
    --trainer.max_steps %MAX_STEPS% ^
    --trainer.val_check_interval %VAL_CHECK_INTERVAL% ^
    --trainer.logger.class_path pytorch_lightning.loggers.TensorBoardLogger ^
    --trainer.logger.init_args.save_dir ..\..\output\experiments ^
    --trainer.logger.init_args.name "%exp_name%" ^
    --model.model_name vit-b16-224-in21k ^
    --model.training_mode %training_mode% ^
    --model.optimizer %optimizer% ^
    --model.lr %lr% ^
    --model.weight_decay %weight_decay% ^
    --model.warmup_steps %warmup_steps% ^
    --model.scheduler cosine ^
    --model.lora_r %lora_r% ^
    --model.lora_alpha %lora_alpha% ^
    --model.lora_target_modules "['query', 'value']" ^
    --data.dataset custom ^
    --data.root "%DATASET_ROOT%" ^
    --data.num_classes %NUM_CLASSES% ^
    --data.size %IMAGE_SIZE% ^
    --data.batch_size %batch_size% ^
    --data.workers 4 ^
    --model_checkpoint.filename "best-step-{step}-{val_acc:.4f}" ^
    --model_checkpoint.monitor val_acc ^
    --model_checkpoint.mode max ^
    --model_checkpoint.save_last true

REM Export TensorBoard logs to CSV
echo 📊 Esportazione TensorBoard logs in CSV...
for /d %%v in (..\..\output\experiments\%exp_name%\version_*) do (
    if exist "%%v" (
        tensorboard --logdir "%%v" --export_to_csv "%%v\metrics.csv" 2>nul || echo ⚠️  TensorBoard export fallito per %%v
    )
)

echo ✅ Completato: %exp_name%
echo ---------------------------------------------------
goto :eof

:test_all_models
for /d %%i in (..\..\output\experiments\*) do (
    if exist "%%i" (
        for /d %%j in ("%%i\version_*") do (
            if exist "%%j\checkpoints" if exist "%%j\config.yaml" (
                for %%k in ("%%j\checkpoints\best-*.ckpt") do (
                    if exist "%%k" (
                        echo 🔍 Testing: %%~nxi
                        python ..\..\main.py test --ckpt_path "%%k" --config "%%j\config.yaml" --trainer.precision 16-mixed
                    )
                )
            )
        )
    )
)
goto :eof 