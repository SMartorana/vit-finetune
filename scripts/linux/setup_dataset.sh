#!/bin/bash

# =============================================================================
# Script per preparare la struttura del dataset custom
# =============================================================================

echo "🚀 Setup Dataset Custom per ViT Fine-tuning"
echo "============================================"

# Configurazione
DATASET_DIR="../../data/custom_dataset"
NUM_CLASSES=10

# Crea struttura directory
echo "📁 Creazione struttura directory..."
mkdir -p "$DATASET_DIR"

# Crea directory per train/val/test splits
for split in train val test; do
    mkdir -p "$DATASET_DIR/$split"
    
    # Crea directory per ogni classe (esempio con 10 classi)
    for ((i=0; i<NUM_CLASSES; i++)); do
        class_name="class_$(printf "%02d" $i)"
        mkdir -p "$DATASET_DIR/$split/$class_name"
        echo "   ✅ Creata: $DATASET_DIR/$split/$class_name"
    done
done

echo ""
echo "📊 Struttura dataset creata:"
echo "../../data/custom_dataset/"
echo "├── train/"
echo "│   ├── class_00/  # <- Inserisci qui le immagini di training"
echo "│   ├── class_01/"
echo "│   ├── ..."
echo "│   └── class_09/"
echo "├── val/"
echo "│   ├── class_00/  # <- Inserisci qui le immagini di validazione"
echo "│   ├── class_01/"
echo "│   ├── ..."
echo "│   └── class_09/"
echo "└── test/"
echo "    ├── class_00/  # <- Inserisci qui le immagini di test"
echo "    ├── class_01/"
echo "    ├── ..."
echo "    └── class_09/"
echo ""

echo "💡 ISTRUZIONI:"
echo "1. Sostituisci 'class_XX' con i nomi reali delle tue classi"
echo "2. Inserisci ~800 immagini per classe in train/"
echo "3. Inserisci ~100 immagini per classe in val/"
echo "4. Inserisci ~100 immagini per classe in test/"
echo "5. Formati supportati: .jpg, .jpeg, .png, .bmp, .tiff"
echo ""

echo "🔧 CONFIGURAZIONE:"
echo "Negli script di training, il dataset viene automaticamente configurato come:"
echo "   DATASET_ROOT=\"$DATASET_DIR\""
echo "   NUM_CLASSES=$NUM_CLASSES"
echo ""

echo "✅ Setup completato!"
echo "🚀 Prossimi passi:"
echo "   1. Popola il dataset con le tue immagini"
echo "   2. Esegui: bash run_experiments.sh"
echo "   3. Analizza risultati: python ../../analyze_results.py" 