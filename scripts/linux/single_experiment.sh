#!/bin/bash

# =============================================================================
# Script per eseguire un singolo esperimento ViT Fine-tuning
# =============================================================================

# Parametri di default (modificabili)
DATASET_ROOT="${1:-../../data/custom_dataset}"
NUM_CLASSES="${2:-10}"
TRAINING_MODE="${3:-full}"        # full, lora, linear
OPTIMIZER="${4:-sgd}"             # sgd, adam, adamw
LR="${5:-0.01}"                   # learning rate
WEIGHT_DECAY="${6:-0.0}"          # weight decay
WARMUP_STEPS="${7:-500}"          # warmup steps
BATCH_SIZE="${8:-128}"            # batch size
LORA_R="${9:-8}"                  # LoRA r (se training_mode=lora)
LORA_ALPHA="${10:-16}"            # LoRA alpha (se training_mode=lora)

# Configurazioni fisse
IMAGE_SIZE=224
MAX_STEPS=5000
VAL_CHECK_INTERVAL=500

# Genera nome esperimento
EXP_NAME="${TRAINING_MODE}_${OPTIMIZER}_lr$(echo $LR | sed 's/\.//g')_wd$(echo $WEIGHT_DECAY | sed 's/\.//g' | sed 's/e-/e/')_w${WARMUP_STEPS}_b${BATCH_SIZE}"

if [ "$TRAINING_MODE" = "lora" ]; then
    EXP_NAME="${EXP_NAME}_r${LORA_R}_a${LORA_ALPHA}"
fi

echo "🚀 Avvio esperimento singolo: $EXP_NAME"
echo "================================================="
echo "Dataset: $DATASET_ROOT"
echo "Classi: $NUM_CLASSES"
echo "Training Mode: $TRAINING_MODE"
echo "Optimizer: $OPTIMIZER"
echo "Learning Rate: $LR"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Warmup Steps: $WARMUP_STEPS"
echo "Batch Size: $BATCH_SIZE"

if [ "$TRAINING_MODE" = "lora" ]; then
    echo "LoRA r: $LORA_R"
    echo "LoRA alpha: $LORA_ALPHA"
fi

echo "================================================="

# Crea directory output
mkdir -p ../../output/single_experiments

# Comando base
CMD="python ../../main.py fit \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --trainer.precision 16-mixed \
    --trainer.max_steps $MAX_STEPS \
    --trainer.val_check_interval $VAL_CHECK_INTERVAL \
    --trainer.logger.class_path pytorch_lightning.loggers.CSVLogger \
    --trainer.logger.init_args.save_dir ../../output/single_experiments \
    --trainer.logger.init_args.name \"$EXP_NAME\" \
    --model.model_name vit-b16-224-in21k \
    --model.training_mode $TRAINING_MODE \
    --model.optimizer $OPTIMIZER \
    --model.lr $LR \
    --model.weight_decay $WEIGHT_DECAY \
    --model.warmup_steps $WARMUP_STEPS \
    --model.scheduler cosine \
    --data.dataset custom \
    --data.root \"$DATASET_ROOT\" \
    --data.num_classes $NUM_CLASSES \
    --data.size $IMAGE_SIZE \
    --data.batch_size $BATCH_SIZE \
    --data.workers 4 \
    --model_checkpoint.filename \"best-step-{step}-{val_acc:.4f}\" \
    --model_checkpoint.monitor val_acc \
    --model_checkpoint.mode max \
    --model_checkpoint.save_last true"

# Aggiungi parametri LoRA se necessario
if [ "$TRAINING_MODE" = "lora" ]; then
    CMD="$CMD \
        --model.lora_r $LORA_R \
        --model.lora_alpha $LORA_ALPHA \
        --model.lora_target_modules \"['query', 'value']\""
fi

echo "Eseguendo comando:"
echo "$CMD"
echo ""

# Esegui comando
eval $CMD

echo ""
echo "✅ Esperimento completato: $EXP_NAME"
echo "📁 Risultati salvati in: ../../output/single_experiments/$EXP_NAME"

# Chiedi se testare il modello
read -p "Vuoi testare il modello appena addestrato? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔍 Avvio test del modello..."
    
    # Trova l'ultimo checkpoint
    LATEST_VERSION=$(ls -d ../../output/single_experiments/$EXP_NAME/version_* 2>/dev/null | sort -V | tail -n 1)
    
    if [ -d "$LATEST_VERSION" ]; then
        CHECKPOINT_DIR="$LATEST_VERSION/checkpoints"
        CONFIG_FILE="$LATEST_VERSION/config.yaml"
        
        if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CONFIG_FILE" ]; then
            BEST_CHECKPOINT=$(ls "$CHECKPOINT_DIR"/best-*.ckpt 2>/dev/null | head -n 1)
            
            if [ -f "$BEST_CHECKPOINT" ]; then
                python ../../main.py test \
                    --ckpt_path "$BEST_CHECKPOINT" \
                    --config "$CONFIG_FILE" \
                    --trainer.precision 16-mixed
                
                echo "✅ Test completato!"
            else
                echo "❌ Checkpoint non trovato in $CHECKPOINT_DIR"
            fi
        else
            echo "❌ File di configurazione non trovato"
        fi
    else
        echo "❌ Directory version non trovata"
    fi
fi

echo ""
echo "📊 Per analizzare tutti i risultati:"
echo "   python ../../analyze_results.py --experiments_dir ../../output/single_experiments" 