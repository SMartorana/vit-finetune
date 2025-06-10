#!/bin/bash

# =============================================================================
# ViT Fine-tuning Experiment Script
# Custom Dataset: ~10 classi, ~1000 immagini per classe
# =============================================================================

# Configurazione dataset custom
DATASET_ROOT="../../data/custom_dataset"  # Modifica con il path del tuo dataset
NUM_CLASSES=10                            # Modifica con il numero delle tue classi
IMAGE_SIZE=224
MAX_STEPS=5000
VAL_CHECK_INTERVAL=500

# Crea directory output se non esiste
mkdir -p ../../output/experiments

# Funzione per eseguire esperimento
run_experiment() {
    local exp_name=$1
    local training_mode=$2
    local lr=$3
    local optimizer=$4
    local weight_decay=$5
    local warmup_steps=$6
    local batch_size=$7
    local lora_r=${8:-16}
    local lora_alpha=${9:-16}
    
    echo "🚀 Avvio esperimento: $exp_name"
    echo "   Training Mode: $training_mode | LR: $lr | Optimizer: $optimizer"
    echo "   Weight Decay: $weight_decay | Warmup: $warmup_steps | Batch: $batch_size"
    
    if [ "$training_mode" = "lora" ]; then
        echo "   LoRA r: $lora_r | LoRA alpha: $lora_alpha"
        
        python ../../main.py fit \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.precision 16-mixed \
            --trainer.max_steps $MAX_STEPS \
            --trainer.val_check_interval $VAL_CHECK_INTERVAL \
            --trainer.logger.class_path pytorch_lightning.loggers.CSVLogger \
            --trainer.logger.init_args.save_dir ../../output/experiments \
            --trainer.logger.init_args.name "$exp_name" \
            --model.model_name vit-b16-224-in21k \
            --model.training_mode $training_mode \
            --model.optimizer $optimizer \
            --model.lr $lr \
            --model.weight_decay $weight_decay \
            --model.warmup_steps $warmup_steps \
            --model.scheduler cosine \
            --model.lora_r $lora_r \
            --model.lora_alpha $lora_alpha \
            --model.lora_target_modules "['query', 'value']" \
            --data.dataset custom \
            --data.root "$DATASET_ROOT" \
            --data.num_classes $NUM_CLASSES \
            --data.size $IMAGE_SIZE \
            --data.batch_size $batch_size \
            --data.workers 4 \
            --model_checkpoint.filename "best-step-{step}-{val_acc:.4f}" \
            --model_checkpoint.monitor val_acc \
            --model_checkpoint.mode max \
            --model_checkpoint.save_last true
    else
        python ../../main.py fit \
            --trainer.accelerator gpu \
            --trainer.devices 1 \
            --trainer.precision 16-mixed \
            --trainer.max_steps $MAX_STEPS \
            --trainer.val_check_interval $VAL_CHECK_INTERVAL \
            --trainer.logger.class_path pytorch_lightning.loggers.CSVLogger \
            --trainer.logger.init_args.save_dir ../../output/experiments \
            --trainer.logger.init_args.name "$exp_name" \
            --model.model_name vit-b16-224-in21k \
            --model.training_mode $training_mode \
            --model.optimizer $optimizer \
            --model.lr $lr \
            --model.weight_decay $weight_decay \
            --model.warmup_steps $warmup_steps \
            --model.scheduler cosine \
            --data.dataset custom \
            --data.root "$DATASET_ROOT" \
            --data.num_classes $NUM_CLASSES \
            --data.size $IMAGE_SIZE \
            --data.batch_size $batch_size \
            --data.workers 4 \
            --model_checkpoint.filename "best-step-{step}-{val_acc:.4f}" \
            --model_checkpoint.monitor val_acc \
            --model_checkpoint.mode max \
            --model_checkpoint.save_last true
    fi
    
    echo "✅ Completato: $exp_name"
    echo "---------------------------------------------------"
}

# Funzione per testare modello
test_model() {
    local exp_name=$1
    local checkpoint_path=$2
    local config_path=$3
    
    echo "🔍 Test del modello: $exp_name"
    
    python ../../main.py test \
        --ckpt_path "$checkpoint_path" \
        --config "$config_path" \
        --trainer.precision 16-mixed
    
    echo "✅ Test completato: $exp_name"
    echo "---------------------------------------------------"
}

echo "🎯 Avvio esperimenti ViT Fine-tuning"
echo "Dataset: $DATASET_ROOT"
echo "Classi: $NUM_CLASSES"
echo "Image Size: $IMAGE_SIZE"
echo "================================================="

# =============================================================================
# ESPERIMENTI FULL FINE-TUNING
# =============================================================================

echo "🔥 FULL FINE-TUNING EXPERIMENTS"

# Full + SGD + Different LRs
run_experiment "full_sgd_lr001_wd0_w500_b128" "full" 0.01 "sgd" 0.0 500 128
run_experiment "full_sgd_lr003_wd0_w500_b128" "full" 0.03 "sgd" 0.0 500 128
run_experiment "full_sgd_lr005_wd0_w500_b128" "full" 0.05 "sgd" 0.0 500 128

# Full + SGD + Weight Decay
run_experiment "full_sgd_lr01_wd1e4_w500_b128" "full" 0.01 "sgd" 1e-4 500 128
run_experiment "full_sgd_lr01_wd1e3_w500_b128" "full" 0.01 "sgd" 1e-3 500 128

# Full + SGD + Different Warmup
run_experiment "full_sgd_lr01_wd0_w100_b128" "full" 0.01 "sgd" 0.0 100 128
run_experiment "full_sgd_lr01_wd0_w1000_b128" "full" 0.01 "sgd" 0.0 1000 128

# Full + AdamW
run_experiment "full_adamw_lr001_wd1e4_w500_b128" "full" 0.001 "adamw" 1e-4 500 128
run_experiment "full_adamw_lr003_wd1e4_w500_b128" "full" 0.003 "adamw" 1e-4 500 128

# Full + Different Batch Sizes
run_experiment "full_sgd_lr01_wd0_w500_b64" "full" 0.01 "sgd" 0.0 500 64
run_experiment "full_sgd_lr01_wd0_w500_b256" "full" 0.01 "sgd" 0.0 500 256

# =============================================================================
# ESPERIMENTI LoRA FINE-TUNING
# =============================================================================

echo "⚡ LoRA FINE-TUNING EXPERIMENTS"

# LoRA + Different r values
run_experiment "lora_r1_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 1 16
run_experiment "lora_r4_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 4 16
run_experiment "lora_r8_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 8 16
run_experiment "lora_r16_a16_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 16 16

# LoRA + Different alpha values
run_experiment "lora_r8_a8_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 8 8
run_experiment "lora_r8_a32_sgd_lr005_wd0_w500_b128" "lora" 0.05 "sgd" 0.0 500 128 8 32

# LoRA + Different LRs
run_experiment "lora_r8_a16_sgd_lr001_wd0_w500_b128" "lora" 0.01 "sgd" 0.0 500 128 8 16
run_experiment "lora_r8_a16_sgd_lr003_wd0_w500_b128" "lora" 0.03 "sgd" 0.0 500 128 8 16
run_experiment "lora_r8_a16_sgd_lr01_wd0_w500_b128" "lora" 0.1 "sgd" 0.0 500 128 8 16

# LoRA + AdamW
run_experiment "lora_r8_a16_adamw_lr001_wd1e4_w500_b128" "lora" 0.001 "adamw" 1e-4 500 128 8 16
run_experiment "lora_r8_a16_adamw_lr003_wd1e4_w500_b128" "lora" 0.003 "adamw" 1e-4 500 128 8 16

# =============================================================================
# ESPERIMENTI LINEAR PROBE
# =============================================================================

echo "📊 LINEAR PROBE EXPERIMENTS"

# Linear + Different LRs
run_experiment "linear_sgd_lr01_wd0_w100_b128" "linear" 0.1 "sgd" 0.0 100 128
run_experiment "linear_sgd_lr05_wd0_w100_b128" "linear" 0.5 "sgd" 0.0 100 128
run_experiment "linear_sgd_lr1_wd0_w100_b128" "linear" 1.0 "sgd" 0.0 100 128

# Linear + AdamW
run_experiment "linear_adamw_lr001_wd1e4_w100_b128" "linear" 0.001 "adamw" 1e-4 100 128
run_experiment "linear_adamw_lr01_wd1e4_w100_b128" "linear" 0.01 "adamw" 1e-4 100 128

echo "🎉 Tutti gli esperimenti completati!"
echo "📁 Risultati salvati in: ../../output/experiments/"
echo "================================================="

# =============================================================================
# TESTING DEI MIGLIORI MODELLI
# =============================================================================

echo "🔍 AVVIO TESTING AUTOMATICO"

# Aspetta input per procedere con il testing
read -p "Vuoi procedere con il testing automatico dei modelli? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Avvio testing dei modelli..."
    
    # Cerca automaticamente i migliori checkpoint
    for exp_dir in ../../output/experiments/*/; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            
            # Trova l'ultimo checkpoint e config
            latest_version=$(ls -d "$exp_dir"version_* 2>/dev/null | sort -V | tail -n 1)
            
            if [ -d "$latest_version" ]; then
                checkpoint_dir="$latest_version/checkpoints"
                config_file="$latest_version/config.yaml"
                
                if [ -d "$checkpoint_dir" ] && [ -f "$config_file" ]; then
                    # Cerca il best checkpoint
                    best_checkpoint=$(ls "$checkpoint_dir"/best-*.ckpt 2>/dev/null | head -n 1)
                    
                    if [ -f "$best_checkpoint" ]; then
                        echo "🔍 Testing: $exp_name"
                        test_model "$exp_name" "$best_checkpoint" "$config_file"
                    fi
                fi
            fi
        fi
    done
    
    echo "✅ Testing completato per tutti i modelli!"
fi

echo "📊 Per analizzare i risultati:"
echo "python ../../analyze_results.py" 