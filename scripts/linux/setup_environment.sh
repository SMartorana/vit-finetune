#!/bin/bash

# =============================================================================
# Script di Setup Automatico Environment ViT Fine-tuning (Linux/macOS)
# =============================================================================

set -e  # Exit on any error

echo "🚀 Setup Automatico Environment ViT Fine-tuning"
echo "================================================="
echo "Questo script creerà l'environment 'vit-training' con tutte le dipendenze necessarie"
echo ""

# Controlla se conda è disponibile
if ! command -v conda &> /dev/null; then
    echo "❌ ERRORE: Conda non trovato nel PATH"
    echo "   Assicurati di aver installato Anaconda/Miniconda e di averlo aggiunto al PATH"
    echo "   Per installare Miniconda:"
    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "   bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo "✅ Conda trovato:"
conda --version

# Inizializza conda per questo script
eval "$(conda shell.bash hook)"

# Controlla se l'environment esiste già
echo ""
echo "🔍 Controllo se l'environment 'vit-training' esiste già..."
if conda env list | grep -q "vit-training"; then
    echo "⚠️  L'environment 'vit-training' esiste già!"
    read -p "Vuoi rimuoverlo e ricrearlo? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Rimozione environment esistente..."
        conda env remove -n vit-training -y
        echo "✅ Environment rimosso"
    else
        echo "⏭️  Installazione delle dipendenze nell'environment esistente..."
        conda activate vit-training
        skip_create=true
    fi
fi

# Crea nuovo environment se necessario
if [ "$skip_create" != true ]; then
    echo ""
    echo "📦 Creazione environment 'vit-training' con Python 3.10..."
    conda create -n vit-training python=3.10 -y
    echo "✅ Environment creato con successo"
fi

echo ""
echo "🔧 Installazione dipendenze..."

# Attiva environment
conda activate vit-training
echo "✅ Environment 'vit-training' attivato"

# Installa PyTorch e dipendenze conda
echo ""
echo "🧠 Installazione PyTorch e dipendenze conda..."
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
echo "✅ PyTorch installato"

echo ""
echo "📊 Installazione librerie scientifiche..."
conda install -c conda-forge numpy==1.24.3 pandas==2.0.1 scipy==1.10.0 -y
echo "✅ Librerie scientifiche installate"

# Installa dipendenze pip
echo ""
echo "📦 Installazione dipendenze Python via pip..."

echo "  - jsonargparse..."
pip install jsonargparse==4.21.1

echo "  - peft..."
pip install peft==0.3.0

echo "  - pytorch_lightning..."
pip install pytorch_lightning[extra]==2.0.2

echo "  - torchmetrics..."
pip install torchmetrics==0.11.4

echo "  - transformers..."
pip install transformers==4.38.0

echo "  - wandb..."
pip install wandb==0.15.2

echo "  - tensorboard..."
pip install tensorboard

echo "✅ Tutte le dipendenze pip installate"

# Verifica installazione
echo ""
echo "🔍 Verifica installazione..."
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import pytorch_lightning; print('✅ Lightning:', pytorch_lightning.__version__)"
python -c "import transformers; print('✅ Transformers:', transformers.__version__)"
python -c "import tensorboard; print('✅ TensorBoard:', tensorboard.__version__)"

echo ""
echo "🎉 SETUP COMPLETATO CON SUCCESSO!"
echo "================================================="
echo "Environment: vit-training"
echo "Python: 3.10"
echo "PyTorch: 2.2.0 (CUDA 11.8)"
echo "Lightning: 2.0.2"
echo "================================================="
echo ""
echo "📋 PROSSIMI PASSI:"
echo "1. Attiva l'environment: conda activate vit-training"
echo "2. Vai al progetto: cd ../../"
echo "3. Esegui esperimenti: ./scripts/linux/run_experiments.sh"
echo ""
echo "💡 TIP: Aggiungi il dataset in data/custom_dataset/"
echo ""

# Rendi lo script eseguibile automaticamente se non lo è già
chmod +x ../../scripts/linux/*.sh

echo "🔧 Scripts resi eseguibili"
echo "Setup completato! 🚀" 