# Swift Shift

A macOS utility that switches window focus between your MacBook display and an external monitor based on where you're looking — no clicking, no dragging the cursor over first.

Look at the MacBook, the MacBook becomes active. Look at the monitor, the monitor becomes active. That's the whole idea.

## Status

Early-stage research prototype. The computer-vision pipeline works end to end (camera → face landmarks → head pose + iris tracking → classifier), but it's not yet wired up to actual window switching on macOS. Right now the project is entirely about getting the gaze signal itself to be reliable — everything else waits until that's solved.

## Why

Working with a laptop and an external monitor means constantly moving the mouse over before you can type or click on the other screen. It's a small tax you pay dozens of times a day. Swift Shift tries to replace "move mouse, click" with "just look at the screen you want."

## How it works (planned)

```
Camera → face landmarks → head pose + iris position
                                  ↓
                    personalized calibration
                                  ↓
                    temporal smoothing / hysteresis
                                  ↓
                       MAC / MONITOR / UNKNOWN
                                  ↓
                      activate the right window
```

Two halves: a Python CV engine that figures out where you're looking, and a native macOS layer (Swift/AppKit) that detects the external monitor, activates windows, and runs the whole thing as a background/menu-bar app that launches at login.

## What's been built so far

- Camera capture + real-time face landmarks via MediaPipe's Face Landmarker
- Head-pose (yaw) estimation via `solvePnP`
- Iris landmark extraction and eye-relative normalized gaze position
- A labeled data collector, two recorded datasets, and a first trained classifier

## The key finding so far

A logistic regression trained on yaw/pitch/roll + iris position hit **98.88%** accuracy on a held-out split of its own training session — but only **63.11%** when evaluated on a completely fresh recording session with no retraining. That gap is the whole story right now: the features (absolute yaw, absolute iris coordinates) drift with posture, sitting position, and camera distance from session to session. A model that memorizes one session's coordinate system doesn't transfer to the next one.

Separately, the original pitch/roll extraction (`cv2.RQDecomp3x3`) was producing physically nonsensical values (pitch around -121° in one session) — a known instability with decomposed Euler angles near certain orientations. Pitch and roll have been dropped for now in favor of a yaw computed directly from the rotation matrix, which is far more stable but still needs a proper validation pass.

Both findings point the same direction: **this isn't a "train a better classifier" problem, it's a "build a gaze representation that's stable across sessions" problem.** The current plan is a short per-session calibration step (look at the MacBook, look at the monitor, ~2-3 seconds each) that re-anchors the features to the user's current setup, rather than relying on a universal threshold trained once and reused forever.

## Facial asymmetry experiment

A useful observation changed the direction of the project: when looking toward the external monitor, one side of the face becomes more occluded from the MacBook camera; looking back at the MacBook, the face reads as more symmetric. Instead of treating this as just an abstract yaw angle, we tested whether that changing facial geometry could be a useful signal on its own.

`cv/face_geometry.py` measures left/right facial asymmetry using nose-to-landmark distances around the eyes, cheeks, jaw, and forehead, boiled down to a single **face ratio**: left-side distance from nose ÷ right-side distance from nose.

| Condition | MacBook | Monitor |
|---|---|---|
| Normal | ~1.0 | ~4.2 |
| Closer | ~1.06–1.07 | ~3.7–3.8 |
| Further | ~1.5–1.6 | ~5.9–6.1 |
| Head slightly left | ~2.5–2.6 | ~2.2–2.3 |
| Head slightly right | ~0.26–0.27 | ~1.8–2.0 |

Under normal viewing conditions the separation is strong. But deliberately tilting the head caused the MacBook and monitor ranges to overlap — face ratio is highly sensitive to head orientation, so it can't work as a standalone universal threshold. It's better thought of as another head-orientation signal to combine with yaw and iris position, not a replacement for either.

## Repo layout

```
SwiftShift/
├── models/               # local MediaPipe Face Landmarker model
├── cv/                   # CV engine: landmarks, head pose, iris, feature experiments,
│                         # data collection, training/evaluation scripts
├── data/                 # recorded gaze datasets (gitignored)
└── macos/                # native macOS layer (not started yet)
```

## Stack

**CV engine:** Python 3.11, OpenCV, MediaPipe, NumPy, scikit-learn
**macOS layer (planned):** Swift, AppKit, CoreGraphics, Accessibility APIs

## Privacy

Everything runs locally. Camera frames are never uploaded or stored beyond what's needed for the current frame. The CV engine is only meant to run while an external monitor is connected — it should stay dormant otherwise to keep CPU usage near zero.

## Right now

Not trying to connect anything to window switching yet. The only goal at this stage is a gaze representation that holds up across sessions — different posture, different day, same accuracy. Everything downstream (calibration UI, smoothing, macOS integration) depends on getting that right first.

## License

TBD