# GazeSwitch 👁️

> **A gaze-controlled display switching utility for macOS.**

GazeSwitch uses computer vision to detect which display you're looking at and automatically shift your active window to that display.

The goal is simple:

**Look at your MacBook → MacBook becomes active.**
**Look at your external monitor → external monitor becomes active.**

No clicking. No moving the mouse to another screen just to interact with it.

---

## Why?

When working with a MacBook connected to an external monitor, switching between displays can become surprisingly annoying.

Normally, you have to:

1. Move the cursor to the other display.
2. Click somewhere on it.
3. Start typing/interacting.

GazeSwitch aims to remove that interaction entirely by using **where you're looking as the intent signal**.

---

## How It Works

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

The application continuously estimates the user's gaze direction and determines which display they are attending to.

---

## Core Features

### 🎯 Gaze-Based Display Switching

Automatically determine whether the user is looking at:

* MacBook display
* External monitor
* Neither / uncertain

### 🖥️ Automatic Monitor Detection

GazeSwitch only activates the computer-vision engine when an external monitor is connected.

```text
Mac starts
    │
    ▼
GazeSwitch running in background
    │
    ▼
External monitor connected?
    │
 ┌──┴──┐
 NO    YES
 │      │
 ▼      ▼
Sleep   Start CV
```

When the monitor is disconnected, camera processing stops automatically.

### ⚡ Fast Switching

Display switching should feel instantaneous enough that the user can simply look at the other screen and start typing.

### 🧠 Calibration

Every desk and camera position is different.

GazeSwitch will therefore support a short calibration process where the user looks at each display so the system can learn their personal gaze geometry.

### 🛡️ Stability / Hysteresis

The system won't switch displays because of a single noisy frame.

A display switch requires the predicted gaze direction to remain stable for a short period.

### ❓ Unknown State

If the system cannot confidently determine where the user is looking, it does nothing.

This prevents accidental switching when:

* Looking away
* Looking between displays
* Looking at a phone
* Face is temporarily lost
* Lighting conditions are poor

### 🔒 Local Processing

Camera frames are processed locally.

No video or gaze data needs to leave the computer.

---

# Architecture

GazeSwitch will use two major components.

```text
                 GazeSwitch
                     │
          ┌──────────┴──────────┐
          │                     │
     CV Engine              macOS Layer
          │                     │
      Python                 Swift
          │                     │
    OpenCV +                AppKit +
    MediaPipe              Accessibility
          │                     │
          └──────────┬──────────┘
                     │
                Communication
```

### Computer Vision Engine

Responsible for:

* Camera capture
* Face detection
* Face landmarks
* Head-pose estimation
* Eye/iris tracking
* Gaze estimation
* Gaze classification
* Confidence estimation

### macOS Layer

Responsible for:

* Detecting connected displays
* Monitoring display configuration changes
* Activating windows
* Moving/restoring cursor position
* Background execution
* Menu-bar application
* Launch at login
* macOS permissions

---

# Technology Stack

## Computer Vision

* Python
* OpenCV
* MediaPipe
* NumPy
* scikit-learn

## macOS

* Swift
* AppKit
* CoreGraphics
* Accessibility APIs

## Communication

The Python CV engine and native macOS controller will communicate through a lightweight local IPC mechanism.

---

# Project Structure

The project will eventually look approximately like this:

```text
GazeSwitch/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── cv/
│   ├── camera.py
│   ├── face_tracker.py
│   ├── head_pose.py
│   ├── eye_gaze.py
│   ├── calibration.py
│   ├── classifier.py
│   └── main.py
│
├── macos/
│   └── GazeSwitch/
│       ├── App.swift
│       ├── DisplayManager.swift
│       ├── WindowManager.swift
│       ├── GazeController.swift
│       └── ...
│
├── config/
│   └── calibration.json
│
├── tests/
│   ├── test_head_pose.py
│   ├── test_gaze.py
│   └── test_classifier.py
│
└── requirements.txt
```

The exact structure may evolve as the project develops.

---

# Development Roadmap

## Phase 1 — Camera & Face Tracking

* [ ] Set up Python environment
* [ ] Access MacBook camera
* [ ] Capture real-time frames
* [ ] Install MediaPipe
* [ ] Detect face landmarks
* [ ] Visualize landmarks
* [ ] Measure FPS

---

## Phase 2 — Head Pose Estimation

* [ ] Extract relevant facial landmarks
* [ ] Estimate yaw
* [ ] Estimate pitch
* [ ] Estimate roll
* [ ] Visualize head direction
* [ ] Test stability under different positions

---

## Phase 3 — Eye Gaze Estimation

* [ ] Extract eye landmarks
* [ ] Track iris position
* [ ] Normalize eye coordinates
* [ ] Combine eye direction with head pose
* [ ] Estimate gaze direction

---

## Phase 4 — Calibration

Create a calibration system:

```text
       Look at MacBook

              ●

       [ Calibrate ]
```

followed by:

```text
       Look at Monitor

              ●

       [ Calibrate ]
```

Store the user's gaze characteristics locally.

---

## Phase 5 — Gaze Classification

Classify gaze into:

```text
MAC
MONITOR
UNKNOWN
```

Initially use geometric rules.

Later experiment with:

* Logistic Regression
* SVM
* Random Forest
* Lightweight neural network

if necessary.

---

## Phase 6 — macOS Integration

* [ ] Detect external displays
* [ ] Detect display connection/disconnection
* [ ] Activate application windows
* [ ] Identify windows associated with each display
* [ ] Restore cursor position
* [ ] Request Accessibility permissions

---

## Phase 7 — Background Application

Turn the project into a proper macOS utility:

* [ ] Menu-bar application
* [ ] Start at login
* [ ] Automatically enable when monitor connects
* [ ] Automatically sleep when monitor disconnects
* [ ] Enable/disable tracking
* [ ] Calibration interface
* [ ] Status indicator

---

# Design Principles

### 1. Local First

No cloud processing.

### 2. Low Latency

The user should be able to look at another display and interact with it almost immediately.

### 3. Stability Over Aggressiveness

It is better to occasionally fail to switch than to constantly switch incorrectly.

### 4. Minimal CPU Usage

When no external monitor is connected, the CV engine should not continuously process camera frames.

### 5. Hardware Agnostic

The system should work with different external monitors and different desk configurations through calibration rather than hard-coded angles.

---

# Future Ideas

Once the basic system works, possible extensions include:

* Multiple external monitors
* Per-monitor calibration
* Cursor teleportation
* Remembering the last active window on each display
* Gaze-controlled application switching
* Dwell-based interaction
* Blink gestures
* Head gestures
* Keyboard shortcuts
* Custom sensitivity
* Menu-bar controls
* Visualization/debug mode
* Automatic calibration assistance

---

# Privacy

GazeSwitch is designed around local processing.

Camera frames should:

* Never be uploaded
* Never be stored by default
* Never be transmitted to a server

The camera is only used while an external display is connected and gaze tracking is active.

---

# Status

🚧 **Early Development**

Current goal:

> Build a reliable prototype that can distinguish between looking at the MacBook display and looking at an external monitor.

---

# License

TBD
