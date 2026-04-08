import scipy.io as sio
import numpy as np
import os
import torch 
from config import (
    WINDOW_SIZE,
    STEP_SIZE,
    TRAIN_REPS,
    PURITY_THRESHOLD,
    BATCH_SIZE,
    NUM_CLASSES
)

class NinaProLoader:

    def __init__(self, file_path, window_size = WINDOW_SIZE , step_size=STEP_SIZE):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.data = None

        # core variables
        self.emg = None

        """
        labels.shape       →  [300000,  1]   ← 2D, but only ONE column
        repetition.shape   →  [300000,  1]   ← same
        labels_flat = self.labels.flatten()      # [300000, 1]  →  [300000]
        reps_flat   = self.repetition.flatten()  # [300000, 1]  →  [300000]

        emg.shape  →  [300000, 12]
        len(emg)   →  300000  
        emg.shape[0]  →  300000    # time samples
        emg.shape[1]  →  12        # channels

        """
        self.labels = None
        self.repetition = None

    def load_mat_file(self):
        print(f'Loading file: {self.file_path}')
        self.data = sio.loadmat(self.file_path)
        print('File loaded successfully.\n')

    def extract_variables(self):
        if self.data is None:
            raise ValueError('Data not loaded. Please call load_mat_file() first.')

        self.emg = self.data.get('emg')                    # main input  [samples x 12]

        restimulus = self.data.get('restimulus')           # ground truth labels (corrected timing)

        if restimulus is not None and np.any(restimulus):
            self.labels = restimulus
            print('Using RESTIMULUS as ground truth labels.')
        else:
            self.labels = self.data.get('stimulus')
            print('RESTIMULUS empty --> falling back to STIMULUS.')

        self.repetition = self.data.get('repetition')      # for train/test split only

    def print_summary(self):
        if self.emg is None:
            raise ValueError('Variables not extracted. Please call extract_variables() first.')

        print('\n===== DATASET SUMMARY =====')
        print(f'File:              {os.path.basename(self.file_path)}')
        print(f'EMG shape:         {self.emg.shape}  -> [time_samples x emg_channels]')
        print(f'Labels shape:      {self.labels.shape}  -> one label per time sample')
        print(f'Repetition shape:  {self.repetition.shape}')

        unique_labels = np.unique(self.labels)
        print(f'\nUnique classes ({len(unique_labels)} total): {unique_labels}')
        print(f'Window size:       {self.window_size} samples ({self.window_size / 2000 * 1000:.0f}ms at 2000Hz)')
        print(f'Step size:         {self.step_size} samples  ({self.step_size / 2000 * 1000:.0f}ms)')
        print('===========================\n')

    def normalise(self, train_reps):
        if self.emg is None:
            raise ValueError('Variables not extracted. Please call extract_variables() first.')

        train_mask = np.isin(self.repetition.flatten(), train_reps) # build a mask for which times samples belong to training repetitions 

        train_emg = self.emg[train_mask]                       # shape: [train_samples x 12]
        self.emg_mean = train_emg.mean(axis = 0)               # shape: [12]
        self.emg_std = train_emg.std(axis = 0)                 # shape: [12]
        

        #print(f'Mean per channel: {self.emg_mean}')   
        #print(f'Std per channel:  {self.emg_std}\n')  

        # then we apply to the entire signal (train + test)
        self.emg = (self.emg - self.emg_mean) / (self.emg_std + 1e-8)   # shape: [time_samples x 12]

        print('Data normalised using training repetitions only.\n')
        # we only use the data of the train to calculate specs because we want to treat the test sample as if its in real time.
        

          


    def create_windows(self, train_reps, purity_threshold):

        if self.emg is None:
            raise ValueError('Variables not extracted. Please call extract_variables() first.')

        labels_flat = self.labels.flatten()
        reps_flat = self.repetition.flatten()

        X_train, Y_train = [], []
        X_test, Y_test = [], []

        for start in range(0, len(self.emg) - self.window_size, self.step_size):

            end = start + self.window_size
            centre_rep = reps_flat[start + self.window_size // 2] 

            if centre_rep == 0:
                continue

            """
           rep 0    → samples recorded between trials, when no repetition 
           was active. These have no meaningful structure.
           → I excluded these entirely from windowing.

           label 0  → rest periods WITHIN a repetition, when the subject 
           was sitting still between movements but the recording 
           was still part of an active trial.
           → these are still in my data.

            """


            # extract the window
            emg_window = self.emg[start:end]
            label_window = labels_flat[start:end]
            


            values, counts = np.unique(label_window, return_counts = True )
            majority_label = values[np.argmax(counts)]
            majority_ratio = counts.max() / self.window_size

            if majority_ratio < purity_threshold:
                continue 
            
            

            if centre_rep in train_reps:
                X_train.append(emg_window)
                Y_train.append(majority_label)
            else:
                X_test.append(emg_window)
                Y_test.append(majority_label)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        print(f'X_train: {X_train.shape}   y_train: {Y_train.shape}')
        print(f'X_test:  {X_test.shape}    y_test:  {Y_test.shape}')
        print(f'Classes in train: {np.unique(Y_train)}')
        print(f'Classes in test:  {np.unique(Y_test)}')

        return X_train, Y_train, X_test, Y_test

    def compute_class_weights(self, y_train, num_classes = NUM_CLASSES):
        
        class_counts = np.array([np.sum(y_train == c) for c in range(num_classes)])
        class_weights = len(y_train) / (num_classes * class_counts )
        class_weights = torch.tensor(class_weights, dtype = torch.float32)

        #print('\nClass weights:')
        #for i, weight in enumerate(class_weights):
            #print(f'  class {i}: {weight:.4f}')
            
        return class_weights


"""
I found that there is a majority class problem in the dataset,
where the rest class dominates the windows by a large margin.

To fix this, we have to implement class weights in the loss function, so that the model
does not predict rest to increase accuracy.
Using this formula:
weight for class i = total_samples / (num_classes * count_of_class_i)

using compute_class_weights (), the model will be penalised 17X more
for getting the rare classes wrong, and we are forcing the model to learn
rather than just ignoring these rare classes.
"""  






if __name__ == '__main__':

    from dataset import create_dataloaders
    
    file_path = os.path.join('..', 'data', 'S1_E1_A1.mat')

    loader = NinaProLoader(file_path, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    loader.load_mat_file()
    loader.extract_variables()
    loader.normalise(TRAIN_REPS)
    loader.print_summary()
    X_train, y_train, X_test, y_test = loader.create_windows(TRAIN_REPS, PURITY_THRESHOLD)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)
    X_batch, y_batch = next(iter(train_loader))
    class_weights = loader.compute_class_weights(y_train, NUM_CLASSES)

    

    

"""
Just for debugging 
print(f'Sample batch:')
    print(f'X_batch shape: {X_batch.shape}')   # expect [32, 200, 12]
    print(f'y_batch shape: {y_batch.shape}')   # expect [32]
    print(f'y_batch values: {y_batch[:8]}')
    print('Class distribution in training set:')


       #print(f'After normalisation:')
    print(f'EMG mean: {loader.emg.mean():.4f}')   # should be ~0
    print(f'EMG std:  {loader.emg.std():.4f}')    # should be ~1
"""