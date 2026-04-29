"""
Batch evaluation script — runs all subjects in data/test/ through the best model
and reports accuracy for each one. No animation, just fast number crunching.
"""
"""HEADS UP: I had to run the test accuracy only on one subject in train.py becaue the RAM did not have enough space to 
allow we to find test accuracy on each subject during training, so I made test_subjects.py that runs the model 
against all the subjects ( both train and test data) to see the general results of my model."""

import torch
import torch.nn as nn
import numpy as np
import os
import time

from model import EMGNet
from model_v2 import EMGNetV2
from data_loader import NinaProLoader
from config import WINDOW_SIZE, TRAIN_REPS, PURITY_THRESHOLD, NUM_CLASSES




MODEL_PATH = 'best_model.pth'

USE_V2     = False
DATA_DIR   = os.path.join('..', 'data', 'test')
STEP_SIZE  = 50


GESTURE_NAMES = {
    0:  'Rest',                       1:  'Thumb up',
    2:  'Index + middle extended',    3:  'Ring + little flexed',
    4:  'Thumb opposing',             5:  'Fingers flexed together',
    6:  'Fist',                       7:  'Pointing index',
    8:  'Adducted fingers',           9:  'Wrist supination (mid)',
    10: 'Wrist pronation (mid)',      11: 'Wrist supination (little)',
    12: 'Wrist pronation (little)',   13: 'Wrist flexion',
    14: 'Wrist extension',            15: 'Wrist radial deviation',
    16: 'Wrist ulnar deviation',      17: 'Wrist extended + fist',
}


def evaluate_subject(file_path, model, device):
   
    filename = os.path.basename(file_path)

    loader = NinaProLoader(file_path)
    loader.load_mat_file()
    loader.extract_variables()
    loader.normalise(train_reps=TRAIN_REPS)

    labels_flat = loader.labels.flatten()
    reps_flat   = loader.repetition.flatten()

    X, y = [], []
    for start in range(0, len(loader.emg) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        centre_rep = reps_flat[start + WINDOW_SIZE // 2]
        if centre_rep == 0:
            continue

        emg_window = loader.emg[start:end]
        acc_window = loader.acc[start:end]
        window = np.concatenate([emg_window, acc_window], axis=1)

        label_window = labels_flat[start:end]
        values, counts = np.unique(label_window, return_counts=True)
        majority_label = values[np.argmax(counts)]
        majority_ratio = counts.max() / WINDOW_SIZE
        if majority_ratio < PURITY_THRESHOLD:
            continue

        X.append(window)
        y.append(majority_label)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        return None

    # run model in batches for speed
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    batch_size = 256
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            predictions = model(batch)
            preds = predictions.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    # compute stats
    correct = (all_preds == y).sum()
    total = len(y)
    overall_acc = correct / total * 100

    per_class_total   = np.zeros(NUM_CLASSES)
    per_class_correct = np.zeros(NUM_CLASSES)
    for cls in range(NUM_CLASSES):
        mask = y == cls
        per_class_total[cls] = mask.sum()
        per_class_correct[cls] = ((all_preds == y) & mask).sum()

    # mean class accuracy (only over classes that appear)
    valid = per_class_total > 0
    class_accs = per_class_correct[valid] / per_class_total[valid] * 100

    return {
        'filename':           filename,
        'total':              total,
        'correct':            correct,
        'overall_acc':        overall_acc,
        'mean_class_acc':     class_accs.mean(),
        'per_class_correct':  per_class_correct,
        'per_class_total':    per_class_total,
    }


def main():
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')

    if not os.path.exists(DATA_DIR):
        print(f'\nERROR: folder {DATA_DIR} does not exist')
        return

    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])
    if not all_files:
        print(f'\nERROR: no .mat files found in {DATA_DIR}')
        return


    print(f'\nLoading {"V2" if USE_V2 else "simple"} model from {MODEL_PATH}...')
    model = EMGNetV2() if USE_V2 else EMGNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f'Model loaded.\n')

   
    print('=' * 70)
    print(f'Running {len(all_files)} subjects...')
    print('=' * 70)

    results = []
    start_time = time.time()

    for i, filename in enumerate(all_files, 1):
        print(f'\n[{i}/{len(all_files)}]  Processing {filename}...', flush=True)
        file_path = os.path.join(DATA_DIR, filename)
        t0 = time.time()

        try:
            result = evaluate_subject(file_path, model, device)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

        elapsed = time.time() - t0
        if result is None:
            print(f'  WARNING: no valid windows')
            continue

        print(f'  Windows: {result["total"]:5d}  |  '
              f'Overall: {result["overall_acc"]:5.2f}%  |  '
              f'Mean class: {result["mean_class_acc"]:5.2f}%  |  '
              f'{elapsed:.1f}s')
        results.append(result)

    total_time = time.time() - start_time
    print(f'\nTotal processing time: {total_time:.1f} seconds\n')

   
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'{"File":<30} {"Windows":>8} {"Overall":>9} {"MeanClass":>10}')
    print('-' * 70)
    for r in results:
        print(f'{r["filename"]:<30} {r["total"]:>8} '
              f'{r["overall_acc"]:>8.2f}% {r["mean_class_acc"]:>9.2f}%')
    print('-' * 70)

    if len(results) > 0:
        avg_overall = np.mean([r['overall_acc'] for r in results])
        avg_mean    = np.mean([r['mean_class_acc'] for r in results])
        print(f'{"AVERAGE":<30} {"":>8} {avg_overall:>8.2f}% {avg_mean:>9.2f}%')

    print('\n\n' + '=' * 70)
    print('PER-CLASS ACCURACY — EACH SUBJECT')
    print('=' * 70)
    header = f'{"Class":<35}'
    for r in results:
        # shorten filename for column
        short = r['filename'].replace('.mat', '').replace('_A1', '')
        header += f'  {short:>8}'
    print(header)
    print('-' * len(header))

    for cls in range(NUM_CLASSES):
        name = GESTURE_NAMES.get(cls, '?')
        row = f'{cls:2d} ({name:28s})  '
        for r in results:
            if r['per_class_total'][cls] > 0:
                acc = r['per_class_correct'][cls] / r['per_class_total'][cls] * 100
                row += f'  {acc:>7.1f}%'
            else:
                row += f'  {"—":>8}'
        print(row)


   
    output_file = 'batch_results.txt'
    with open(output_file, 'w') as f:
        f.write('Batch evaluation results\n')
        f.write(f'Model: {"V2" if USE_V2 else "simple"}\n')
        f.write(f'Files processed: {len(results)}\n')
        f.write(f'Device: {device}\n\n')

        f.write(f'{"File":<30} {"Windows":>8} {"Overall":>9} {"MeanClass":>10}\n')
        f.write('-' * 70 + '\n')
        for r in results:
            f.write(f'{r["filename"]:<30} {r["total"]:>8} '
                    f'{r["overall_acc"]:>8.2f}% {r["mean_class_acc"]:>9.2f}%\n')
        f.write('-' * 70 + '\n')
        if len(results) > 0:
            f.write(f'{"AVERAGE":<30} {"":>8} {avg_overall:>8.2f}% {avg_mean:>9.2f}%\n')

        f.write('\n\nPer-class accuracy per subject:\n\n')
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        for cls in range(NUM_CLASSES):
            name = GESTURE_NAMES.get(cls, '?')
            row = f'{cls:2d} ({name:28s})  '
            for r in results:
                if r['per_class_total'][cls] > 0:
                    acc = r['per_class_correct'][cls] / r['per_class_total'][cls] * 100
                    row += f'  {acc:>7.1f}%'
                else:
                    row += f'  {"—":>8}'
            f.write(row + '\n')

    print(f'\n\nResults saved to {output_file}')


if __name__ == '__main__':
    main()