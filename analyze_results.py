#!/usr/bin/env python3
"""
Script per analizzare i risultati degli esperimenti di fine-tuning ViT
Analizza i file CSV di log e identifica i modelli con le migliori performance
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse


def parse_experiment_name(exp_name):
    """
    Parsing del nome dell'esperimento per estrarre gli iperparametri
    Esempio: 'full_sgd_lr001_wd0_w500_b128' -> {'mode': 'full', 'optimizer': 'sgd', 'lr': 0.01, ...}
    """
    parts = exp_name.split('_')
    params = {}
    
    # Training mode
    params['training_mode'] = parts[0]
    
    # Optimizer
    if len(parts) > 1:
        params['optimizer'] = parts[1]
    
    # Parse remaining parameters
    for part in parts[2:]:
        if part.startswith('lr'):
            # Convert lr001 -> 0.01
            lr_val = part[2:]
            if lr_val == '001':
                params['lr'] = 0.01
            elif lr_val == '003':
                params['lr'] = 0.03
            elif lr_val == '005':
                params['lr'] = 0.05
            elif lr_val == '01':
                params['lr'] = 0.1
            elif lr_val == '1':
                params['lr'] = 1.0
            else:
                params['lr'] = float(lr_val)
        elif part.startswith('wd'):
            wd_val = part[2:]
            if wd_val == '0':
                params['weight_decay'] = 0.0
            elif wd_val == '1e4':
                params['weight_decay'] = 1e-4
            elif wd_val == '1e3':
                params['weight_decay'] = 1e-3
            else:
                params['weight_decay'] = float(wd_val)
        elif part.startswith('w'):
            params['warmup_steps'] = int(part[1:])
        elif part.startswith('b'):
            params['batch_size'] = int(part[1:])
        elif part.startswith('r'):
            params['lora_r'] = int(part[1:])
        elif part.startswith('a') and params['training_mode'] == 'lora':
            params['lora_alpha'] = int(part[1:])
    
    return params


def load_experiment_results(experiments_dir):
    """
    Carica i risultati di tutti gli esperimenti
    """
    results = []
    experiments_path = Path(experiments_dir)
    
    if not experiments_path.exists():
        print(f"❌ Directory {experiments_dir} non trovata!")
        return pd.DataFrame()
    
    print(f"🔍 Scansione directory: {experiments_dir}")
    
    for exp_dir in experiments_path.iterdir():
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        print(f"   📁 Analizzando: {exp_name}")
        
        # Cerca le versioni dell'esperimento
        version_dirs = list(exp_dir.glob("version_*"))
        
        for version_dir in version_dirs:
            metrics_file = version_dir / "metrics.csv"
            
            if metrics_file.exists():
                try:
                    # Carica metriche
                    df = pd.read_csv(metrics_file)
                    
                    if not df.empty and 'val_acc' in df.columns:
                        # Prendi la migliore accuracy di validazione
                        best_val_acc = df['val_acc'].max()
                        best_idx = df['val_acc'].idxmax()
                        
                        # Estrai altri parametri
                        params = parse_experiment_name(exp_name)
                        
                        result = {
                            'experiment_name': exp_name,
                            'version': version_dir.name,
                            'best_val_acc': best_val_acc,
                            'final_step': df.loc[best_idx, 'step'] if 'step' in df.columns else None,
                            **params
                        }
                        
                        # Aggiungi test accuracy se disponibile
                        if 'test_acc' in df.columns:
                            test_acc = df['test_acc'].dropna()
                            if not test_acc.empty:
                                result['test_acc'] = test_acc.iloc[-1]
                        
                        results.append(result)
                        print(f"      ✅ Val Acc: {best_val_acc:.4f}")
                        
                except Exception as e:
                    print(f"      ❌ Errore nel caricare {metrics_file}: {e}")
            else:
                print(f"      ⚠️  File metrics.csv non trovato in {version_dir}")
    
    return pd.DataFrame(results)


def analyze_results(df):
    """
    Analizza i risultati e genera insights
    """
    if df.empty:
        print("❌ Nessun risultato trovato!")
        return
    
    print("\n" + "="*70)
    print("📊 ANALISI RISULTATI ESPERIMENTI ViT FINE-TUNING")
    print("="*70)
    
    # Statistiche generali
    print(f"\n📈 STATISTICHE GENERALI:")
    print(f"   • Esperimenti totali: {len(df)}")
    print(f"   • Migliore Val Accuracy: {df['best_val_acc'].max():.4f}")
    print(f"   • Media Val Accuracy: {df['best_val_acc'].mean():.4f}")
    print(f"   • Deviazione Standard: {df['best_val_acc'].std():.4f}")
    
    # Top 10 esperimenti
    print(f"\n🏆 TOP 10 ESPERIMENTI:")
    top_10 = df.nlargest(10, 'best_val_acc')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"   {i:2d}. {row['experiment_name']:40s} | Val Acc: {row['best_val_acc']:.4f}")
    
    # Analisi per training mode
    print(f"\n🔥 ANALISI PER TRAINING MODE:")
    for mode in df['training_mode'].unique():
        mode_df = df[df['training_mode'] == mode]
        best_mode = mode_df.loc[mode_df['best_val_acc'].idxmax()]
        print(f"   • {mode.upper():12s}: Best = {best_mode['best_val_acc']:.4f} | "
              f"Media = {mode_df['best_val_acc'].mean():.4f} | "
              f"Esperimenti = {len(mode_df)}")
    
    # Analisi per optimizer
    if 'optimizer' in df.columns:
        print(f"\n⚡ ANALISI PER OPTIMIZER:")
        for opt in df['optimizer'].unique():
            opt_df = df[df['optimizer'] == opt]
            best_opt = opt_df.loc[opt_df['best_val_acc'].idxmax()]
            print(f"   • {opt.upper():8s}: Best = {best_opt['best_val_acc']:.4f} | "
                  f"Media = {opt_df['best_val_acc'].mean():.4f} | "
                  f"Esperimenti = {len(opt_df)}")
    
    # Analisi LoRA specifica
    lora_df = df[df['training_mode'] == 'lora']
    if not lora_df.empty and 'lora_r' in lora_df.columns:
        print(f"\n🔧 ANALISI LoRA SPECIFICA:")
        for r in sorted(lora_df['lora_r'].unique()):
            r_df = lora_df[lora_df['lora_r'] == r]
            best_r = r_df.loc[r_df['best_val_acc'].idxmax()]
            print(f"   • r={r:2d}: Best = {best_r['best_val_acc']:.4f} | "
                  f"Media = {r_df['best_val_acc'].mean():.4f} | "
                  f"Esperimenti = {len(r_df)}")
    
    return df


def create_visualizations(df, output_dir="output/analysis"):
    """
    Crea visualizzazioni dei risultati
    """
    if df.empty:
        return
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Accuracy per training mode
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='training_mode', y='best_val_acc')
    plt.title('Validation Accuracy per Training Mode')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Training Mode')
    
    # 2. Accuracy per optimizer
    if 'optimizer' in df.columns:
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='optimizer', y='best_val_acc')
        plt.title('Validation Accuracy per Optimizer')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Optimizer')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap per LoRA parameters
    lora_df = df[df['training_mode'] == 'lora']
    if not lora_df.empty and 'lora_r' in lora_df.columns and 'lora_alpha' in lora_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Crea pivot table
        pivot_table = lora_df.pivot_table(
            values='best_val_acc', 
            index='lora_r', 
            columns='lora_alpha', 
            aggfunc='max'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd')
        plt.title('LoRA Hyperparameter Grid Search Results')
        plt.ylabel('LoRA r')
        plt.xlabel('LoRA alpha')
        
        plt.savefig(f"{output_dir}/lora_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Learning rate analysis
    if 'lr' in df.columns:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        lr_analysis = df.groupby(['training_mode', 'lr'])['best_val_acc'].max().reset_index()
        
        for mode in lr_analysis['training_mode'].unique():
            mode_data = lr_analysis[lr_analysis['training_mode'] == mode]
            plt.plot(mode_data['lr'], mode_data['best_val_acc'], 
                    marker='o', label=mode, linewidth=2, markersize=8)
        
        plt.xlabel('Learning Rate')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Learning Rate vs Performance')
        plt.legend()
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # 5. Batch size analysis
        if 'batch_size' in df.columns:
            plt.subplot(1, 2, 2)
            batch_analysis = df.groupby(['training_mode', 'batch_size'])['best_val_acc'].max().reset_index()
            
            for mode in batch_analysis['training_mode'].unique():
                mode_data = batch_analysis[batch_analysis['training_mode'] == mode]
                plt.plot(mode_data['batch_size'], mode_data['best_val_acc'], 
                        marker='s', label=mode, linewidth=2, markersize=8)
            
            plt.xlabel('Batch Size')
            plt.ylabel('Best Validation Accuracy')
            plt.title('Batch Size vs Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hyperparameter_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Visualizzazioni salvate in: {output_dir}")


def export_results(df, output_dir="output/analysis"):
    """
    Esporta i risultati in formato CSV e Excel
    """
    if df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva risultati completi
    df.to_csv(f"{output_dir}/all_results.csv", index=False)
    
    # Salva top 10
    top_10 = df.nlargest(10, 'best_val_acc')
    top_10.to_csv(f"{output_dir}/top_10_results.csv", index=False)
    
    # Salva summary per training mode
    summary = df.groupby('training_mode')['best_val_acc'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    summary.to_csv(f"{output_dir}/summary_by_mode.csv")
    
    print(f"📄 Risultati esportati in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analizza risultati esperimenti ViT fine-tuning')
    parser.add_argument('--experiments_dir', default='output/experiments', 
                       help='Directory contenente gli esperimenti')
    parser.add_argument('--output_dir', default='output/analysis', 
                       help='Directory per salvare le analisi')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Non generare grafici')
    
    args = parser.parse_args()
    
    # Carica risultati
    print("🚀 Avvio analisi risultati...")
    df = load_experiment_results(args.experiments_dir)
    
    if df.empty:
        print("❌ Nessun risultato trovato!")
        return
    
    # Analizza risultati
    analyze_results(df)
    
    # Crea visualizzazioni
    if not args.no_plots:
        try:
            create_visualizations(df, args.output_dir)
        except ImportError as e:
            print(f"⚠️  Impossibile creare grafici: {e}")
            print("   Installa matplotlib e seaborn: pip install matplotlib seaborn")
    
    # Esporta risultati
    export_results(df, args.output_dir)
    
    print("\n✅ Analisi completata!")
    print(f"📁 Risultati salvati in: {args.output_dir}")


if __name__ == "__main__":
    main() 