# 🚀 Guida Completa agli Esperimenti ViT Fine-tuning

Questa guida ti aiuterà a eseguire esperimenti completi di fine-tuning di Vision Transformers sul tuo dataset custom.

## 📁 Organizzazione Script Multi-Piattaforma

**IMPORTANTE**: Gli script sono ora organizzati per piattaforma nella directory `scripts/`:

```
scripts/
├── linux/          # Script Bash per Linux/macOS/Unix
│   ├── run_experiments.sh
│   ├── single_experiment.sh  
│   └── setup_dataset.sh
├── windows/         # Script Batch per Windows
│   ├── run_experiments.bat
│   ├── single_experiment.bat
│   └── setup_dataset.bat
└── README.md        # Guida rapida degli script
```

**🐧 Linux/macOS:** Usa `scripts/linux/`  
**🪟 Windows:** Usa `scripts/windows/`

## 📋 Prerequisiti

1. **Hardware**: GPU NVIDIA con almeno 8GB VRAM
2. **Software**: Python 3.8+, CUDA, PyTorch
3. **Dataset**: ~10 classi, ~1000 immagini per classe organizzate in format ImageFolder

## 🛠️ Setup Iniziale

### 1. Installa Dipendenze
```cmd
conda create -n vit-training python=3.10
conda activate vit-training
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge numpy==1.24.3 pandas==2.0.1 scipy==1.10.0
pip install jsonargparse==4.21.1
pip install peft==0.3.0
pip install pytorch_lightning[extra]==2.0.2
pip install torchmetrics==0.11.4
pip install transformers==4.38.0
pip install wandb==0.15.2
pip install tensorboard
```

### 1b. Testa l'environment
```python
import torch
print(f"PyTorch version: {torch.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```


### 2. Prepara la Struttura del Dataset
**Linux/macOS:**
```bash
cd scripts/linux
chmod +x *.sh
./setup_dataset.sh
```

**Windows:**
```batch
cd scripts\windows
setup_dataset.bat
```

### 3. Organizza le Tue Immagini
Sposta le tue immagini nella struttura creata:
```
data/custom_dataset/
├── train/
│   ├── classe_1/    # ~800 immagini
│   ├── classe_2/    # ~800 immagini
│   └── ...
├── val/
│   ├── classe_1/    # ~100 immagini
│   ├── classe_2/    # ~100 immagini
│   └── ...
└── test/
    ├── classe_1/    # ~100 immagini
    ├── classe_2/    # ~100 immagini
    └── ...
```

## 🎯 Strategie di Esperimenti

### Strategia 1: Esperimenti Completi (Raccomandato)
Esegue tutti gli esperimenti con diverse combinazioni di iperparametri:

**Linux/macOS:**
```bash
cd scripts/linux
./run_experiments.sh
```

**Windows:**
```batch
cd scripts\windows
run_experiments.bat
```

**Cosa Include:**
- **25+ esperimenti** diversi
- **3 modalità di training**: Full fine-tuning, LoRA, Linear probe
- **Variazioni di iperparametri**: Learning rate, optimizer, weight decay, batch size
- **Testing automatico** dei modelli
- **Analisi comparativa** dei risultati

### Strategia 2: Esperimento Singolo
Per testare rapidamente una configurazione specifica:

**Linux/macOS:**
```bash
cd scripts/linux

# Esempio 1: Full fine-tuning con SGD
./single_experiment.sh ../../data/custom_dataset 10 full sgd 0.01 0.0 500 128

# Esempio 2: LoRA fine-tuning
./single_experiment.sh ../../data/custom_dataset 10 lora sgd 0.05 0.0 500 128 8 16

# Esempio 3: Linear probe
./single_experiment.sh ../../data/custom_dataset 10 linear sgd 0.1 0.0 100 128
```

**Windows:**
```batch
cd scripts\windows

REM Esempio 1: Full fine-tuning con SGD
single_experiment.bat ..\..\data\custom_dataset 10 full sgd 0.01 0.0 500 128

REM Esempio 2: LoRA fine-tuning
single_experiment.bat ..\..\data\custom_dataset 10 lora sgd 0.05 0.0 500 128 8 16

REM Esempio 3: Linear probe  
single_experiment.bat ..\..\data\custom_dataset 10 linear sgd 0.1 0.0 100 128
```

## 📊 Combinazioni di Iperparametri Testati

### Full Fine-tuning
| Parametro | Valori Testati |
|-----------|----------------|
| Learning Rate | 0.01, 0.03, 0.05 |
| Optimizer | SGD, AdamW |
| Weight Decay | 0.0, 1e-4, 1e-3 |
| Warmup Steps | 100, 500, 1000 |
| Batch Size | 64, 128, 256 |

### LoRA Fine-tuning
| Parametro | Valori Testati |
|-----------|----------------|
| LoRA r | 1, 4, 8, 16 |
| LoRA alpha | 8, 16, 32 |
| Learning Rate | 0.01, 0.03, 0.05, 0.1 |
| Optimizer | SGD, AdamW |

### Linear Probe
| Parametro | Valori Testati |
|-----------|----------------|
| Learning Rate | 0.1, 0.5, 1.0 |
| Optimizer | SGD, AdamW |
| Warmup Steps | 100 |

## 🏷️ Nomenclatura dei Modelli

I modelli sono salvati con nomi che indicano chiaramente gli iperparametri:

**Format**: `{mode}_{optimizer}_lr{lr}_wd{wd}_w{warmup}_b{batch}[_r{lora_r}_a{lora_alpha}]`

**Esempi:**
- `full_sgd_lr001_wd0_w500_b128` → Full fine-tuning, SGD, LR=0.01, no weight decay
- `lora_r8_a16_sgd_lr005_wd0_w500_b128` → LoRA r=8, alpha=16, SGD, LR=0.05
- `linear_sgd_lr1_wd0_w100_b128` → Linear probe, SGD, LR=1.0

## 📈 Analisi dei Risultati

### Analisi Automatica
```bash
python analyze_results.py
```

**Output:**
- 📊 **Statistiche generali**: Media, best, deviazione standard
- 🏆 **Top 10 esperimenti**: Migliori performance
- 🔥 **Analisi per modalità**: Full vs LoRA vs Linear
- ⚡ **Analisi per optimizer**: SGD vs AdamW
- 🔧 **Analisi LoRA**: Impatto di r e alpha
- 📈 **Grafici**: Boxplot, heatmap, curves
- 📄 **Export CSV**: Risultati dettagliati

### Analisi Manuale
```bash
# Trova i migliori modelli
find output/experiments -name "*.ckpt" | grep "best-" | sort

# Controlla log specifico
cat output/experiments/full_sgd_lr001_wd0_w500_b128/version_0/metrics.csv
```

## 🎛️ Comandi Avanzati

### Test di un Modello Specifico
```bash
python main.py test \
    --ckpt_path output/experiments/EXPERIMENT_NAME/version_0/checkpoints/best-*.ckpt \
    --config output/experiments/EXPERIMENT_NAME/version_0/config.yaml \
    --trainer.precision 16-mixed
```

### Riprendere un Training
```bash
python main.py fit \
    --config output/experiments/EXPERIMENT_NAME/version_0/config.yaml \
    --ckpt_path output/experiments/EXPERIMENT_NAME/version_0/checkpoints/last.ckpt
```

### Modifica Configurazione al Volo
```bash
python main.py fit \
    --config configs/full/cifar100.yaml \
    --data.root data/custom_dataset \
    --data.num_classes 10 \
    --model.lr 0.05 \
    --trainer.max_steps 3000
```

## 📋 Checklist Esperimenti

### Prima di Iniziare
- [ ] Dataset preparato e organizzato
- [ ] GPU disponibile e funzionante
- [ ] Spazio disco sufficiente (almeno 10GB liberi)
- [ ] Piattaforma corretta selezionata (Linux vs Windows)

### Durante gli Esperimenti
- [ ] Monitorare utilizzo GPU: `nvidia-smi`
- [ ] Controllare log per errori
- [ ] Verificare salvataggio checkpoint

### Dopo gli Esperimenti
- [ ] Eseguire analisi risultati
- [ ] Identificare migliore configurazione
- [ ] Testare modello finale
- [ ] Salvare best model per produzione

## 🚨 Troubleshooting

### Errori Comuni

**GPU Out of Memory**
```bash
# Linux
./single_experiment.sh ../../data/custom_dataset 10 full sgd 0.01 0.0 500 64

# Windows  
single_experiment.bat ..\..\data\custom_dataset 10 full sgd 0.01 0.0 500 64
```

**Dataset Non Trovato**
```bash
# Linux
ls -la ../../data/custom_dataset/train/

# Windows
dir ..\..\data\custom_dataset\train\
```

**Mancano Dipendenze**
```bash
pip install pytorch-lightning transformers peft
```

### Performance Tips

1. **Usa LoRA** se hai limitazioni di memoria/tempo
2. **Inizia con Linear Probe** per baseline veloce  
3. **Monitora overfitting** con validation curves
4. **Usa early stopping** se validation accuracy plateaus

## 📚 Interpretazione Risultati

### Accuracy Target
- **Excellent**: >95% validation accuracy
- **Good**: 90-95% validation accuracy  
- **Acceptable**: 85-90% validation accuracy
- **Poor**: <85% validation accuracy

### Selezione Miglior Modello
1. **Highest validation accuracy**
2. **Stable training** (no oscillazioni)
3. **Good generalization** (val_acc ≈ test_acc)
4. **Efficiency** (LoRA vs Full quando performance simili)

## 🎯 Best Practices

1. **Start Small**: Testa prima con pochi steps
2. **Monitor Early**: Controlla prime epoche per overfitting
3. **Save Everything**: Mantieni tutti i checkpoint importanti
4. **Compare Systematically**: Usa naming convention consistent
5. **Document Results**: Annota insights e osservazioni

## 🔄 Cross-Platform Compatibility

Gli script sono 100% compatibili tra piattaforme:
- **Path relativi** funzionano su entrambi i sistemi
- **Risultati identici** indipendentemente dalla piattaforma
- **Stessa nomenclatura** dei modelli
- **Output compatibili** per l'analisi

---

**🚀 Happy Fine-tuning!** 

Per domande o problemi, controlla i log dettagliati in `output/experiments/` o consulta la documentazione PyTorch Lightning. 