import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
import os
import sys

from model import EMGNet
from model_v2 import EMGNetV2
from data_loader import NinaProLoader
from config import WINDOW_SIZE, TRAIN_REPS, PURITY_THRESHOLD, NUM_CLASSES



MODEL_PATH = 'best_model.pth'  
USE_V2     = False
DATA_DIR   = os.path.join('..', 'data', 'test')


GESTURE_NAMES = {
    0:  'Rest',
    1:  'Thumb up',
    2:  'Index + middle extended',
    3:  'Ring + little flexed',
    4:  'Thumb opposing',
    5:  'Fingers flexed together',
    6:  'Fist',
    7:  'Pointing index',
    8:  'Adducted fingers',
    9:  'Wrist supination (mid)',
    10: 'Wrist pronation (mid)',
    11: 'Wrist supination (little)',
    12: 'Wrist pronation (little)',
    13: 'Wrist flexion',
    14: 'Wrist extension',
    15: 'Wrist radial deviation',
    16: 'Wrist ulnar deviation',
    17: 'Wrist extended + fist',
}



def get_finger_states(gesture_class):
    state = {
        'thumb': False, 'index': False, 'middle': False,
        'ring': False, 'little': False, 'wrist_angle': 0,
    }

    if gesture_class == 0:
        state['thumb'] = state['index'] = state['middle'] = True
        state['ring'] = state['little'] = True
    elif gesture_class == 1:
        state['index'] = state['middle'] = state['ring'] = state['little'] = True
    elif gesture_class == 2:
        state['thumb'] = state['ring'] = state['little'] = True
    elif gesture_class == 3:
        state['ring'] = state['little'] = True
    elif gesture_class == 5:
        state['index'] = state['middle'] = state['ring'] = state['little'] = True
    elif gesture_class == 6:
        state['thumb'] = state['index'] = state['middle'] = True
        state['ring'] = state['little'] = True
    elif gesture_class == 7:
        state['thumb'] = state['middle'] = state['ring'] = state['little'] = True
    elif gesture_class == 13:
        state['wrist_angle'] = -30
    elif gesture_class == 14:
        state['wrist_angle'] = 30
    elif gesture_class == 15:
        state['wrist_angle'] = 15
    elif gesture_class == 16:
        state['wrist_angle'] = -15
    elif gesture_class == 17:
        state['thumb'] = state['index'] = state['middle'] = True
        state['ring'] = state['little'] = True
        state['wrist_angle'] = 30

    return state


def draw_hand(ax, gesture_class, label, is_correct=None):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    state = get_finger_states(gesture_class)

    if is_correct is None:
        color = '#E8C4A0'
    elif is_correct:
        color = '#90EE90'
    else:
        color = '#FF9999'

    rotation = Affine2D().rotate_deg(state['wrist_angle']) + ax.transData

    palm = patches.Rectangle((-0.8, 0), 1.6, 1.5, color=color, transform=rotation)
    ax.add_patch(palm)

    finger_data = [
        ('thumb',  -1.0, 0.8, 0.8),
        ('index',  -0.6, 1.5, 1.2),
        ('middle', -0.2, 1.5, 1.3),
        ('ring',    0.2, 1.5, 1.2),
        ('little',  0.6, 1.5, 1.0),
    ]

    for name, x, y, full_length in finger_data:
        length = 0.3 if state[name] else full_length
        finger = patches.Rectangle((x-0.1, y), 0.2, length, color=color, transform=rotation)
        ax.add_patch(finger)

    ax.set_title(f'{label}\n{GESTURE_NAMES.get(gesture_class, "Unknown")}',
                 fontsize=11, pad=10)



def classify_subject_type(filename):
    """Guess subject type from filename conventions."""
    fname = filename.lower()
    # DB3 (amputees) often named S1_A1_E1 or similar with 'A' 
    # DB2 (able-bodied) typically S1_E1_A1
    if '_a' in fname and fname.index('_a') < fname.find('_e') if '_e' in fname else True:
        # amputee files typically have _A before _E
        pass
    return 'unknown'


def choose_subject():
    print('\n' + '='*50)
    print('      HAND GESTURE VISUALISER')
    print('='*50)

    # check data directory
    if not os.path.exists(DATA_DIR):
        print(f'\nERROR: folder {DATA_DIR} does not exist')
        print(f'Please create it and add the .mat files')
        sys.exit(1)

    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])
    if not all_files:
        print(f'\nERROR: no .mat files found in {DATA_DIR}')
        sys.exit(1)

    # show all available files
    print(f'\nAvailable files in {DATA_DIR}:')
    print('-' * 50)
    for i, f in enumerate(all_files, 1):
        print(f'  {i:2d}. {f}')
    print('-' * 50)

    # pick file
    while True:
        try:
            selection = input(f'\nSelect file number (1-{len(all_files)}): ').strip()
            idx = int(selection)
            if 1 <= idx <= len(all_files):
                chosen_file = all_files[idx - 1]
                break
        except ValueError:
            pass
        print(f'Invalid input. Please enter a number between 1 and {len(all_files)}.')

    # ask for subject type for labelling
    print('\nWhat type of subject is this file?')
    print('  1 - Able-bodied')
    print('  2 - Amputee')

    while True:
        choice = input('\nEnter choice (1 or 2): ').strip()
        if choice in ('1', '2'):
            subject_type = 'able-bodied' if choice == '1' else 'amputee'
            break
        print('Invalid input, try 1 or 2.')

    full_path = os.path.join(DATA_DIR, chosen_file)
    print(f'\n{"-"*50}')
    print(f'Selected: {chosen_file}')
    print(f'Type:     {subject_type}')
    print(f'{"-"*50}\n')
    return full_path, subject_type, chosen_file


def run_visualisation():
    file_path, subject_type, filename = choose_subject()

    # load model
    print('Loading trained model...')
    if USE_V2:
        model = EMGNetV2()
    else:
        model = EMGNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print('Model loaded.\n')

    # load data
    loader = NinaProLoader(file_path)
    loader.load_mat_file()
    loader.extract_variables()
    loader.normalise(train_reps=TRAIN_REPS)

    # use ALL windows, not just test reps
    X_all, y_all = [], []

    labels_flat = loader.labels.flatten()
    reps_flat = loader.repetition.flatten()

    for start in range(0, len(loader.emg) - WINDOW_SIZE, 50):
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

        X_all.append(window)
        y_all.append(majority_label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f'Loaded {len(X_all)} windows from {filename}\n')

    # setup figure
    fig = plt.figure(figsize=(14, 7))
    ax_pred = fig.add_subplot(1, 3, 1)
    ax_true = fig.add_subplot(1, 3, 2)
    ax_conf = fig.add_subplot(1, 3, 3)

    stats = {'total': 0, 'correct': 0,
             'per_class_correct': np.zeros(NUM_CLASSES),
             'per_class_total':   np.zeros(NUM_CLASSES)}

    def update(frame):
        window = X_all[frame]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predictions = model(x)
            probabilities = torch.softmax(predictions, dim=1).squeeze().numpy()
            predicted_class = int(predictions.argmax(dim=1).item())

        true_class = int(y_all[frame])
        is_correct = predicted_class == true_class

        stats['total'] += 1
        stats['per_class_total'][true_class] += 1
        if is_correct:
            stats['correct'] += 1
            stats['per_class_correct'][true_class] += 1

        running_acc = stats['correct'] / stats['total'] * 100

        draw_hand(ax_pred, predicted_class, 'PREDICTED', is_correct)
        draw_hand(ax_true, true_class, 'TRUE', None)

        ax_conf.clear()
        colors = ['#4CAF50' if i == predicted_class else '#B0B0B0'
                  for i in range(NUM_CLASSES)]
        ax_conf.barh(range(NUM_CLASSES), probabilities, color=colors)
        ax_conf.set_yticks(range(NUM_CLASSES))
        ax_conf.set_yticklabels([f'{i}' for i in range(NUM_CLASSES)], fontsize=8)
        ax_conf.set_xlim(0, 1)
        ax_conf.set_xlabel('Confidence')
        ax_conf.set_title('Class probabilities', fontsize=11)
        ax_conf.invert_yaxis()

        fig.suptitle(
            f'{subject_type.title()} Subject ({filename})  |  '
            f'Window {frame+1}/{len(X_all)}  |  '
            f'Running accuracy: {running_acc:.1f}% ({stats["correct"]}/{stats["total"]})',
            fontsize=12, fontweight='bold'
        )

    ani = FuncAnimation(fig, update, frames=len(X_all),
                        interval=50, repeat=False)

    plt.tight_layout()
    plt.show()

   
    print('\n\n' + '='*50)
    print('         FINAL ANALYSIS')
    print('='*50)
    print(f'Subject type:     {subject_type}')
    print(f'File:             {filename}')
    print(f'Total windows:    {stats["total"]}')
    print(f'Correct:          {stats["correct"]}')
    print(f'Overall accuracy: {stats["correct"]/stats["total"]*100:.2f}%')

    print('\n=== Per-class accuracy ===')
    for cls in range(NUM_CLASSES):
        if stats['per_class_total'][cls] > 0:
            acc = stats['per_class_correct'][cls] / stats['per_class_total'][cls] * 100
            name = GESTURE_NAMES.get(cls, '?')
            print(f'  class {cls:2d} ({name:28s}): {acc:6.2f}%  '
                  f'({int(stats["per_class_correct"][cls])}/{int(stats["per_class_total"][cls])})')

    valid = stats['per_class_total'] > 0
    class_accs = stats['per_class_correct'][valid] / stats['per_class_total'][valid] * 100
    print(f'\nMean class accuracy: {class_accs.mean():.2f}%')
    print(f'Std class accuracy:  {class_accs.std():.2f}%')


if __name__ == '__main__':
    run_visualisation()