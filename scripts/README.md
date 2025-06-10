# 📁 Scripts Directory

Questa directory contiene gli script organizzati per entrambe le piattaforme:

```
scripts/
├── linux/          # Script Bash per sistemi Unix/Linux/macOS
│   ├── run_experiments.sh
│   ├── single_experiment.sh
│   └── setup_dataset.sh
└── windows/         # Script Batch per Windows
    ├── run_experiments.bat
    ├── single_experiment.bat
    └── setup_dataset.bat
```

## 🚀 Utilizzo Rapido

### 🐧 Linux/macOS/Unix:
```bash
cd scripts/linux
chmod +x *.sh
./setup_dataset.sh              # Setup iniziale
./run_experiments.sh            # Tutti gli esperimenti
./single_experiment.sh [args]   # Esperimento singolo
```

### 🪟 Windows:
```batch
cd scripts\windows
setup_dataset.bat               # Setup iniziale
run_experiments.bat             # Tutti gli esperimenti  
single_experiment.bat [args]    # Esperimento singolo
```

## 📖 Documentazione Completa

Per la **guida completa agli esperimenti**, consulta:
**[📖 EXPERIMENTS_README.md](../EXPERIMENTS_README.md)**

Contiene:
- Setup dettagliato
- Spiegazione degli iperparametri  
- Analisi dei risultati
- Troubleshooting
- Best practices

---

**💡 Questo README è solo per la struttura degli script. Per tutto il resto, usa la guida principale!** 