#!/usr/bin/env python3
"""
Script per spostare e organizzare immagini dal dataset sorgente al custom dataset.
Suddivide automaticamente le immagini in train (80%), val (10%), test (10%).
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple
import glob

# Formati di immagine supportati
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}

def get_image_files(directory: Path) -> List[Path]:
    """
    Trova tutti i file immagine in una directory in modo ricorsivo.
    Esplora tutte le sottocartelle fino a quando non ci sono più directory.
    
    Args:
        directory: Path della directory da esplorare
        
    Returns:
        Lista di Path dei file immagine trovati ricorsivamente
    """
    image_files = []
    
    def scan_directory_recursive(current_dir: Path):
        """Funzione ricorsiva per esplorare directory e sottodirectory."""
        try:
            for item in current_dir.iterdir():
                if item.is_file() and item.suffix in SUPPORTED_FORMATS:
                    image_files.append(item)
                elif item.is_dir():
                    # Ricorsione per esplorare sottocartelle
                    scan_directory_recursive(item)
        except PermissionError:
            print(f"⚠️  Permessi negati per accedere a: {current_dir}")
        except Exception as e:
            print(f"⚠️  Errore nell'esplorare {current_dir}: {e}")
    
    scan_directory_recursive(directory)
    return image_files

def split_images(image_files: List[Path], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Suddivide una lista di immagini in train, validation e test set.
    
    Args:
        image_files: Lista di file immagine
        train_ratio: Percentuale per training (default: 0.8)
        val_ratio: Percentuale per validation (default: 0.1)  
        test_ratio: Percentuale per test (default: 0.1)
        
    Returns:
        Tuple contenente (train_files, val_files, test_files)
    """
    # Verifica che le percentuali sommino a 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Le percentuali devono sommare a 1.0, trovato: {total_ratio}")
    
    # Mescola casualmente la lista
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)
    
    total_files = len(shuffled_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    # Suddivisione
    train_files = shuffled_files[:train_count]
    val_files = shuffled_files[train_count:train_count + val_count]
    test_files = shuffled_files[train_count + val_count:]
    
    return train_files, val_files, test_files

def create_dataset_structure(custom_dataset_path: Path, class_names: List[str]) -> None:
    """
    Crea la struttura del custom dataset con i nomi delle classi reali.
    
    Args:
        custom_dataset_path: Path del custom dataset
        class_names: Lista dei nomi delle classi
    """
    print(f"📁 Creazione struttura custom dataset in: {custom_dataset_path}")
    
    # Crea directory principale se non esiste
    custom_dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Crea struttura train/val/test con classi reali
    for split in ['train', 'val', 'test']:
        split_dir = custom_dataset_path / split
        split_dir.mkdir(exist_ok=True)
        
        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            print(f"   ✅ Creata: {class_dir}")

def move_images(source_files: List[Path], destination_dir: Path) -> int:
    """
    Sposta i file immagine dalla sorgente alla destinazione.
    
    Args:
        source_files: Lista di file sorgente
        destination_dir: Directory di destinazione
        
    Returns:
        Numero di file spostati con successo
    """
    moved_count = 0
    
    for source_file in source_files:
        try:
            destination_file = destination_dir / source_file.name
            
            # Evita sovrascritture accidentali
            if destination_file.exists():
                base_name = source_file.stem
                extension = source_file.suffix
                counter = 1
                while destination_file.exists():
                    destination_file = destination_dir / f"{base_name}_{counter}{extension}"
                    counter += 1
            
            shutil.move(str(source_file), str(destination_file))
            moved_count += 1
            
        except Exception as e:
            print(f"❌ Errore spostando {source_file}: {e}")
    
    return moved_count

def process_dataset(source_path: Path, custom_dataset_path: Path) -> None:
    """
    Processa il dataset sorgente e lo organizza nel custom dataset.
    
    Args:
        source_path: Path del dataset sorgente
        custom_dataset_path: Path del custom dataset di destinazione
    """
    print(f"🚀 Inizio elaborazione dataset")
    print(f"📂 Sorgente: {source_path}")
    print(f"📁 Destinazione: {custom_dataset_path}")
    print("=" * 60)
    
    # Trova tutte le directory delle classi nel dataset sorgente
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"Nessuna directory di classe trovata in: {source_path}")
    
    class_names = [d.name for d in class_dirs]
    print(f"🏷️  Classi trovate ({len(class_names)}): {', '.join(class_names)}")
    
    # Crea struttura custom dataset
    create_dataset_structure(custom_dataset_path, class_names)
    
    # Statistiche globali
    total_moved = {'train': 0, 'val': 0, 'test': 0}
    
    # Processa ogni classe
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n📸 Elaborazione classe: {class_name}")
        
        # Trova tutte le immagini nella classe (ricorsivamente)
        image_files = get_image_files(class_dir)
        
        if not image_files:
            print(f"⚠️  Nessuna immagine trovata in {class_dir}")
            continue
        
        # Mostra statistiche sulla profondità della ricerca
        subdirs_found = set()
        for img_file in image_files:
            relative_path = img_file.relative_to(class_dir)
            if len(relative_path.parts) > 1:  # Ha sottocartelle
                subdirs_found.add(relative_path.parent)
        
        print(f"   📊 Trovate {len(image_files)} immagini")
        if subdirs_found:
            print(f"   📁 Sottocartelle esplorate: {len(subdirs_found)} ({', '.join(str(s) for s in sorted(subdirs_found)[:3])}{'...' if len(subdirs_found) > 3 else ''})")
        
        # Suddividi le immagini
        train_files, val_files, test_files = split_images(image_files)
        
        print(f"   📈 Suddivisione: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Sposta le immagini nei rispettivi dataset
        splits = {
            'train': train_files,
            'val': val_files, 
            'test': test_files
        }
        
        for split_name, files in splits.items():
            if files:
                destination_dir = custom_dataset_path / split_name / class_name
                moved_count = move_images(files, destination_dir)
                total_moved[split_name] += moved_count
                print(f"   ✅ {split_name}: {moved_count}/{len(files)} immagini spostate")
    
    # Riepilogo finale
    print("\n" + "=" * 60)
    print("📊 RIEPILOGO FINALE:")
    print(f"🎓 Training:   {total_moved['train']} immagini")
    print(f"🔍 Validation: {total_moved['val']} immagini")
    print(f"🧪 Test:       {total_moved['test']} immagini")
    print(f"📈 Totale:     {sum(total_moved.values())} immagini")
    print("✅ Elaborazione completata!")

def main():
    """Funzione principale del script."""
    parser = argparse.ArgumentParser(
        description="Sposta e organizza immagini dal dataset sorgente al custom dataset per ViT fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python data_parser.py /path/to/source/dataset
  python data_parser.py C:\\Users\\Data\\MyDataset --custom-path ..\\data\\my_custom_dataset
  python data_parser.py /path/to/source --seed 42
        """
    )
    
    parser.add_argument(
        'source_path',
        type=str,
        help='Path del dataset sorgente contenente le directory delle classi'
    )
    
    parser.add_argument(
        '--custom-path',
        type=str,
        default='../data/custom_dataset',
        help='Path del custom dataset di destinazione (default: ../data/custom_dataset)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed per la randomizzazione (default: 42)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Percentuale per training set (default: 0.8)'
    )
    
    parser.add_argument(
        '--val-ratio', 
        type=float,
        default=0.1,
        help='Percentuale per validation set (default: 0.1)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float, 
        default=0.1,
        help='Percentuale per test set (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Imposta seed per riproducibilità
    random.seed(args.seed)
    
    # Converte i path in oggetti Path
    source_path = Path(args.source_path)
    custom_dataset_path = Path(args.custom_path)
    
    # Validazioni
    if not source_path.exists():
        print(f"❌ Errore: Il path sorgente non esiste: {source_path}")
        return 1
    
    if not source_path.is_dir():
        print(f"❌ Errore: Il path sorgente non è una directory: {source_path}")
        return 1
    
    # Verifica percentuali
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Errore: Le percentuali devono sommare a 1.0, trovato: {total_ratio}")
        return 1
    
    try:
        # Processa il dataset
        process_dataset(source_path, custom_dataset_path)
        return 0
        
    except Exception as e:
        print(f"❌ Errore durante l'elaborazione: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
