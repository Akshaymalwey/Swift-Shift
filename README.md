# GazeSwitch 👁️

> **A gaze-controlled macOS utility that automatically switches focus between displays based on where you're looking.**

GazeSwitch is a computer-vision project that aims to remove the friction of switching between a MacBook display and an external monitor.

The idea is simple:

**Look at the MacBook → MacBook becomes active.**
**Look at the external monitor → external monitor becomes active.**

No clicking. No manually moving the cursor to activate the other display.

---

# Project Status

🚧 **Early Development / Research Prototype**

The computer-vision pipeline is currently being developed and evaluated.

We have successfully:

* Captured the MacBook camera feed.
* Detected facial landmarks using MediaPipe.
* Extracted eye/iris landmarks.
* Estimated head orientation.
* Extracted gaze-related features.
* Collected labeled gaze datasets.
* Trained an initial gaze classifier.
* Tested the classifier on a completely unseen recording session.
* Identified important limitations in the initial feature representation.

The current system is **not yet connected to macOS window switching**.

---

# Motivation

When working with a MacBook connected to an external monitor, switching between the two displays can become repetitive:

1. Move the cursor to the other display.
2. Click somewhere on it.
3. Start typing/interacting.

GazeSwitch attempts to make this automatic.

Instead of using the mouse to communicate:

> "I want to interact with this display."

the user's gaze becomes the signal.

---

# Initial Architecture

The original planned architecture is:

```text
                 MacBook Camera
                       │
                       ▼
              ┌─────────────────┐
              │  Face Detection │
              │ + Face Landmarks│
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Head Pose          Eye Gaze
         Estimation         Estimation
              │                 │
              └────────┬────────┘
                       ▼
                Gaze Classifier
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           MAC      MONITOR    UNKNOWN
             │         │
             ▼         ▼
       Activate Mac  Activate
         Window      Monitor Window
```

The architecture is intentionally divided into two major components.

### Computer Vision Engine

Responsible for:

* Camera capture
* Face landmarks
* Head-pose estimation
* Eye/iris tracking
* Gaze feature extraction
* Gaze classification
* Confidence estimation

### macOS Layer

Responsible for:

* Detecting connected displays
* Monitoring display connection/disconnection
* Activating windows
* Cursor management
* Background execution
* Menu-bar application
* Launch at login
* Accessibility permissions

---

# Technology Stack

## Computer Vision

* Python 3.11
* OpenCV
* MediaPipe
* NumPy
* scikit-learn
* joblib

## Planned macOS Layer

* Swift
* AppKit
* CoreGraphics
* Accessibility APIs

---

# Current Project Structure

```text
GazeSwitch/
│
├── README.md
├── .gitignore
├── .python-version
├── requirements.txt
│
├── models/
│   └── face_landmarker.task
│
├── cv/
│   ├── camera_test.py
│   ├── head_pose.py
│   ├── eye_landmarks.py
│   ├── eye_gaze.py
│   ├── collect_gaze_data.py
│   ├── analyze_gaze_data.py
│   ├── find_threshold.py
│   ├── train_gaze_model.py
│   ├── evaluate_gaze_model.py
│   └── compare_datasets.py
│
├── data/
│   ├── gaze_samples.csv
│   └── gaze_samples_test.csv
│
├── tests/
│
├── macos/
│
└── config/
```

The structure will evolve as the project moves from experimentation toward the actual application.

---

# Phase 1 — Environment Setup

The project is being developed on an **Apple Silicon M3 Mac**.

The system architecture was verified as:

```text
arm64
```

Python 3.11.9 was selected for the project using `pyenv`.

A project-specific virtual environment was created:

```text
.venv/
```

Core dependencies were installed:

```text
opencv-python
mediapipe
numpy
scikit-learn
joblib
```

The environment successfully runs the MediaPipe computer-vision pipeline.

---

# Phase 2 — Camera & Face Tracking

The first milestone was to verify that the MacBook camera could be accessed and processed in real time.

Pipeline:

```text
MacBook Camera
      ↓
OpenCV
      ↓
MediaPipe Face Landmarker
      ↓
Facial Landmarks
```

The MediaPipe Face Landmarker model is stored locally:

```text
models/face_landmarker.task
```

The system successfully tracks facial landmarks in real time.

This established the foundation for all subsequent gaze experiments.

---

# Phase 3 — Head Pose Estimation

The next step was estimating the orientation of the user's head.

Relevant quantities:

```text
Yaw   → horizontal head direction
Pitch → vertical head direction
Roll  → head tilt
```

The initial implementation used facial landmarks and OpenCV's `solvePnP()`.

Conceptually:

```text
Facial landmarks
       +
Approximate 3D face model
       ↓
cv2.solvePnP()
       ↓
Rotation
       ↓
Head orientation
```

## Initial Experiment

A manual experiment produced approximately:

```text
Looking at MacBook  → ~0°
Looking at Monitor  → ~70°
Extreme Left        → ~80°
Extreme Right       → ~20°
```

This suggested that head orientation could provide a strong signal for distinguishing the two displays.

However, subsequent dataset collection revealed that the exact values were dependent on posture and session conditions.

---

# Phase 4 — Eye / Iris Tracking

MediaPipe Face Landmarker also provides detailed eye and iris landmarks.

The project extracts iris landmarks and calculates an eye-relative normalized position.

Conceptually:

```text
Eye boundaries
      +
Iris center
      ↓
Normalized iris position
      ↓
X / Y gaze features
```

The normalized coordinates are approximately:

```text
0.0 → one side
0.5 → center
1.0 → opposite side
```

This prevents raw pixel coordinates from being directly dependent on camera resolution.

---

# Phase 5 — Initial Gaze Dataset

A labeled data collector was created.

The collector records:

```text
timestamp
yaw
pitch
roll

left_iris_x
left_iris_y

right_iris_x
right_iris_y

avg_iris_x
avg_iris_y

target
```

The target is one of:

```text
mac
monitor
```

Samples are collected at approximately 10 samples/second.

---

## Dataset 1

The first dataset contained:

```text
MacBook:  223 samples
Monitor:  222 samples
Total:    445 samples
```

This dataset was used for initial analysis and model development.

---

# Phase 6 — Initial Data Analysis

The first analysis compared the distributions of:

* Head yaw
* Iris X position
* Yaw vs. iris X

The results showed that **yaw was the strongest individual feature**.

A simple yaw threshold was tested.

The best threshold found was approximately:

```text
Yaw threshold ≈ -38°
```

with:

```text
Accuracy ≈ 93.26%
```

This established a useful baseline.

However, 93.26% was not considered sufficient for directly controlling window focus because even occasional incorrect switches would make the system frustrating to use.

---

# Phase 7 — First Machine Learning Model

A Logistic Regression classifier was trained using:

```text
yaw
pitch
roll
left_iris_x
left_iris_y
right_iris_x
right_iris_y
```

The model was implemented using a scikit-learn pipeline:

```text
Feature Scaling
      ↓
StandardScaler
      ↓
Logistic Regression
```

On a random 80/20 split of Dataset 1, the model achieved:

```text
Accuracy: 98.88%
```

The confusion matrix was:

```text
                 Predicted
                 Mac   Monitor

Actual Mac       45      0
Actual Monitor    1     43
```

This looked extremely promising.

However, the random split was potentially optimistic because consecutive video frames are highly correlated.

Therefore, a completely separate test session was collected.

---

# Phase 8 — Fresh Dataset Evaluation

A second dataset was collected in a new session.

```text
MacBook:  237 samples
Monitor:  251 samples
Total:    488 samples
```

The previously trained model was **not retrained** on this dataset.

Instead:

```text
Dataset 1
   ↓
Train model
   ↓
Saved model
   ↓
Dataset 2
   ↓
Evaluate
```

The result was:

```text
Accuracy: 63.11%
```

Confusion matrix:

```text
                 Predicted
                 Mac   Monitor

Actual Mac       178     59

Actual Monitor   121    130
```

This was a major finding.

The initial 98.88% accuracy did **not** generalize to a completely new session.

---

# What We Learned

The 63.11% result revealed that the problem is not simply:

> "We need a more powerful classifier."

Instead, the current features are too dependent on the exact recording conditions.

In particular, the distributions shifted between sessions.

### Average iris X

```text
              Dataset 1     Dataset 2

MacBook          0.482          0.477
Monitor          0.549          0.503
```

The monitor gaze distribution moved substantially closer to the MacBook distribution.

### Yaw

```text
              Dataset 1     Dataset 2

MacBook        -17.54        -25.77
Monitor        -41.99        -34.09
```

The separation between the two targets also became smaller.

This explains why the classifier struggled on the fresh session.

---

# Important Bug Discovered

The collected pitch and roll values showed extremely large and unstable values in some sessions.

For example:

```text
Dataset 1 Mac pitch mean      ≈ -121°
Dataset 2 Mac pitch mean      ≈ 12°

Dataset 1 Monitor roll mean   ≈ 100°
Dataset 2 Monitor roll mean   ≈ 25°
```

These values are physically suspicious for normal head movement.

This suggests that the current Euler-angle extraction using `cv2.RQDecomp3x3()` needs to be revisited.

Therefore:

> **Pitch and roll should not currently be trusted as stable features.**

The next implementation step is to improve the head-pose calculation and establish a reliable yaw representation.

---

# Current Understanding of the Problem

The project has moved from:

```text
"Train a classifier to recognize Mac vs Monitor"
```

toward:

```text
"Build a personalized gaze/display calibration system"
```

This is an important architectural change.

A universal model using absolute gaze coordinates may not be sufficiently robust because the measurements depend on:

* Sitting position
* Head position
* Camera position
* Distance from the camera
* Screen geometry
* Where on each screen the user is looking
* Session-to-session variation

---

# Planned Calibration Approach

Instead of assuming fixed global thresholds, GazeSwitch will eventually calibrate itself for the current user and setup.

When an external monitor is detected:

```text
Monitor Connected
       ↓
Start Calibration
       ↓
Look at MacBook
       ↓
Collect samples
       ↓
Look at Monitor
       ↓
Collect samples
       ↓
Build personalized decision boundary
```

The calibration can be short, for example:

```text
MacBook → ~2–3 seconds
Monitor → ~2–3 seconds
```

The system can then learn the current relationship between the user's gaze and the two displays.

This should be substantially more robust than relying on a universal hard-coded threshold.

---

# Revised CV Architecture

The current thinking is:

```text
                 Camera
                   │
                   ▼
          MediaPipe Face Landmarker
                   │
          ┌────────┴────────┐
          ▼                 ▼
      Head Pose          Eye / Iris
          │                 │
          │                 │
          └────────┬────────┘
                   ▼
          Relative Gaze Features
                   │
                   ▼
          Personalized Calibration
                   │
                   ▼
             Gaze Decision
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
        MAC     MONITOR   UNKNOWN
```

Head pose will provide the coarse direction, while eye/iris information can help resolve ambiguous cases.

---

# Stability Requirements

Even a highly accurate classifier should not immediately switch windows on every frame.

The final system will require temporal stability:

```text
Raw predictions
       ↓
Temporal smoothing
       ↓
Confidence threshold
       ↓
Hysteresis / debounce
       ↓
Stable target
       ↓
Switch window
```

For example:

```text
MONITOR
MONITOR
MONITOR
MONITOR
MONITOR
        ↓
Stable MONITOR
        ↓
Activate monitor window
```

while:

```text
MONITOR
MAC
MONITOR
MAC
MONITOR
        ↓
Uncertain
        ↓
Do nothing
```

An `UNKNOWN` state will also be supported.

---

# Planned macOS Integration

Once the CV system is reliable, the next major component will be native macOS integration.

The final application should:

### Detect displays

```text
MacBook only
     ↓
CV engine sleeping
```

and:

```text
MacBook + External Monitor
     ↓
Start gaze tracking
```

### Detect disconnection

```text
External monitor disconnected
          ↓
Stop camera processing
          ↓
GazeSwitch remains in background
```

### Activate windows

```text
Gaze → MacBook
      ↓
Activate MacBook window
```

and:

```text
Gaze → Monitor
      ↓
Activate monitor window
```

### Background operation

The eventual application should live as a lightweight macOS menu-bar utility.

---

# Privacy

The project is designed around local processing.

Camera frames should:

* Never be uploaded
* Never be transmitted
* Never be stored by default

Computer vision should only run while gaze tracking is active.

When no external monitor is connected, the CV engine should sleep to minimize CPU usage.

---

# Development Roadmap

## Completed

* [x] GitHub repository
* [x] Python 3.11 environment
* [x] ARM64/M3 development environment
* [x] OpenCV setup
* [x] MediaPipe setup
* [x] Face Landmarker model
* [x] Real-time face tracking
* [x] Head-pose prototype
* [x] Iris landmark extraction
* [x] Normalized iris features
* [x] Labeled gaze data collector
* [x] Initial dataset collection
* [x] Initial data visualization
* [x] Yaw-only baseline
* [x] Logistic Regression baseline
* [x] Fresh-session evaluation
* [x] Identification of session-dependent feature drift

## Current

* [ ] Fix and validate head-pose calculation
* [ ] Establish stable yaw representation
* [ ] Improve eye-relative gaze representation
* [ ] Design personalized calibration
* [ ] Collect post-calibration data
* [ ] Evaluate cross-session robustness
* [ ] Implement `MAC / MONITOR / UNKNOWN`
* [ ] Add temporal smoothing
* [ ] Add confidence threshold
* [ ] Add hysteresis/debounce

## Future

* [ ] macOS display detection
* [ ] Automatic CV activation when monitor connects
* [ ] Automatic CV shutdown when monitor disconnects
* [ ] macOS window activation
* [ ] Cursor position restoration
* [ ] Swift menu-bar application
* [ ] Launch at login
* [ ] Accessibility permissions
* [ ] Multiple external monitor support
* [ ] Per-monitor calibration
* [ ] Configuration interface

---

# Current Experimental Results

| Experiment                          |     Result |
| ----------------------------------- | ---------: |
| Yaw-only threshold                  | **93.26%** |
| Logistic Regression — random split  | **98.88%** |
| Logistic Regression — fresh session | **63.11%** |

The large gap between random-split and fresh-session performance is currently the most important result in the project.

It demonstrates that **cross-session robustness is the primary computer-vision challenge** before integrating the system with macOS.

---

# Current Goal

The immediate goal is **not** to connect the classifier to window switching.

The immediate goal is:

> **Build a gaze representation that remains reliable when the user's posture and recording session change.**

Only after that is solved will the project move to automatic macOS display and window control.

---

# License

TBD
