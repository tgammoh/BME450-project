# BME450-project
EMG Hand Movement Classification Using Neural Networks

## Team members 
Tala Gammoh 

### Project description 
This project focuses on the classification of hand movements using electromyography (EMG)
signals from the Ninapro Database. The goal is to develop a machine learning pipeline that
can recognize different hand gestures based on muscle activity recorded from the forearm.

Signals from Subject 1, Exercise 1 are used as the primary data source. The raw EMG signals 
are loaded from MATLAB .mat files and preprocessed to prepare them for training a neural network model.

The final objective is to train a neural network that can automatically learn features from EMG 
signals and accurately classify different hand movements. This type of system has important 
applications in prosthetic control, human–machine interfaces, and rehabilitation technologies.

---

## Dataset

The project uses **NinaPro DB2** — a publicly available surface EMG dataset designed for prosthetic research.

| Property | Value |
|---|---|
| Signal type | 12-channel surface EMG + 36-channel accelerometer |
| Sampling rate | 2000 Hz |
| Number of gestures | 18 (0 = rest, 1-17 = different hand/wrist movements) |
| Repetitions per gesture | 6 (4 used for training, 2 for testing) |
| Training subjects | 5 able-bodied subjects (S1, S2, S3, S5, S6) |
| Test subjects | held-out reps of S1, plus unseen subjects for cross-subject evaluation |
| Window size | 200 samples (100 ms) |
| Window step | 50 samples (25 ms) |

The accelerometer data comes from 12 Delsys Trigno sensors, each with a 3-axis accelerometer co-located with its EMG electrode. Combining EMG with accelerometer as a unified 48-channel input was the single biggest improvement in the project (+10.47% accuracy over EMG alone).



## Usage

### Install dependencies

```bash
pip install torch numpy scipy scikit-learn matplotlib tensorboard
```

### Train the model

```bash
cd src
python3 train.py
```

Trains on 5 subjects, saves the best checkpoint to `best_model.pth`, and logs to TensorBoard.



### Evaluate on unseen subjects and trained subjects

```bash
python3 test_subjects.py
```

Runs the trained model on every file in `data/test/`. Reports overall accuracy, mean per-class accuracy, and a per-class breakdown table comparing all subjects.

### Real-time prediction demo

```bash
python3 hand_visualiser.py
```

Interactive demo that:
- lets you pick a test file
- animates predicted vs true hand gesture in real time
- shows confidence bars across all 18 classes
- reports running accuracy

### View training curves

```bash
tensorboard --logdir src/runs/
```

Opens an interactive dashboard showing loss and accuracy curves over epochs.

---

## Future work

- **More training subjects** — NinaPro DB2 has 40 subjects total; using more of them should force the model to learn universal features instead of memorising individuals.
- **Transfer learning + per-user fine-tuning** — pre-train on a large subject pool, then adapt with 1-2 repetitions from each new user (how commercial prosthetics actually work).
- **Real-time hardware integration** — couple this classifier to a physical sensor array and motor system for a functional prosthetic demo.
- **Interactive deployment as a web app** — package the model into a Streamlit app for easier demonstration and sharing.

---

## References

- Atzori, M. et al. (2014). *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*. Scientific Data. [NinaPro dataset paper]
- Zhang, S. et al. (2023). *Transfer Learning Enhanced Cross-Subject Hand Gesture Recognition with sEMG*. Journal of Medical and Biological Engineering.
- Krasoulis, A. et al. (2017). *Improved prosthetic hand control with concurrent use of myoelectric and inertial measurements*. Journal of NeuroEngineering and Rehabilitation.
