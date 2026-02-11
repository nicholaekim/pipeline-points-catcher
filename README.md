# StretchSense XR Glove Tracker

Real-time hand tracking visualization system for StretchSense XR gloves using OSC protocol.

## Features

- **Live Joint Display** - View all 26 hand joint values for both left and right hands in real-time
- **3D Hand Visualization** - Animated 3D skeleton view showing hand movements
- **Recording & Playback** - Record hand movements and play them back as animations
- **CSV Export** - Save hand pose data to CSV files
- **Calibration** - Zero out initial hand positions with one click

## Installation

```bash
pip install numpy matplotlib python-osc
```

## Usage

```bash
python run.py
```

The program listens on OSC ports 9000-9005 for glove data.

---

## Step-by-Step Operation Guide

### Step 1: Start the Program
1. Open a terminal/command prompt
2. Navigate to the tracker folder
3. Run `python run.py`
4. The main window will open showing "OpenXR Hand Joints - Left & Right"

### Step 2: Connect Your Gloves
1. Turn on your StretchSense XR gloves
2. Make sure they are configured to send OSC data to your computer's IP on ports 9000-9005
3. Once connected, the device names will appear at the top (e.g., "Left: reality glove (l)")
4. The joint values in the table will start updating with live data
5. The packet counter at the bottom shows how many data packets have been received

### Step 3: Calibrate (Recommended)
1. Put your hands in a neutral/flat position
2. Click **"Recalibrate"** button
3. This sets the current position as the zero point
4. All joint values will reset relative to this new baseline

### Step 4: View Live 3D Visualization
1. Make sure gloves are connected and receiving data
2. Click **"Live 3D View"** button
3. A new window opens with two 3D plots (left hand and right hand)
4. Red dots show joint positions (numbered 0-25)
5. Green/blue lines connect the joints to form the finger skeleton
6. Move your hands to see the skeleton animate in real-time
7. You can click and drag to rotate the 3D view

### Step 5: Record Hand Movement
1. Click **"Record"** button (turns red and says "Stop")
2. Move your hands to capture the movement you want
3. Click **"Stop"** to end recording
4. The console will show how many frames were captured

### Step 6: Playback Recording
1. After recording, click **"Playback"** button
2. A new window opens showing your recorded movement as a looping animation
3. The frame counter shows current position in the recording

### Step 7: Export Data to CSV
1. Click **"Export to CSV"** button
2. Choose where to save the file
3. The file contains all 26 joint positions for both hands
4. Useful for data analysis or sharing pose snapshots

---

## What Each Button Does

| Button | What It Does |
|--------|--------------|
| **Recalibrate** | Resets the zero position for all joints. Use this when you first put on the gloves or if the values seem off. The current hand position becomes the new "zero" baseline. |
| **Export to CSV** | Saves a snapshot of the current hand pose to a CSV file. The file includes all 26 joint X/Y/Z values for both left and right hands with a timestamp in the filename. |
| **Live 3D View** | Opens a real-time animated 3D visualization showing both hands as skeleton models. The joints move as you move your physical hands. Updates at ~60 FPS. |
| **Record** | Starts capturing frames of hand movement data. Click again to stop. Each frame stores the position of all joints for both hands. |
| **Playback** | Plays back your recorded hand movement as a looping 3D animation. Must record something first. Shows frame-by-frame animation of the captured movement. |

---

## Understanding the Display

### Main Window Joint Table
- **Idx**: Joint index number (0-25)
- **Joint Name**: Anatomical name of the joint
- **Left X/Y/Z**: Position values for left hand (orange text)
- **Right X/Y/Z**: Position values for right hand (blue text)

### 3D Visualization
- **Red dots**: Individual joint positions
- **Numbers (0-25)**: Joint index labels
- **Green lines**: Skeleton connections (left hand)
- **Blue lines**: Skeleton connections (right hand)
- **X/Y/Z axes**: 3D coordinate system

### Packet Counter
- Shows "Packets: L=### R=###"
- L = number of packets received from left glove
- R = number of packets received from right glove
- If these numbers aren't increasing, check glove connection

---

## Controls Summary

| Button | Function |
|--------|----------|
| Recalibrate | Reset hand position calibration |
| Export to CSV | Save current pose to file |
| Live 3D View | Open real-time 3D hand skeleton |
| Record | Start/stop recording movement |
| Playback | Play recorded animation |

## Hand Joint Structure

Tracks 26 joints per hand (OpenXR standard):

- **Palm & Wrist** (0-1)
- **Thumb** (2-5): metacarpal → tip
- **Index** (6-10): metacarpal → tip
- **Middle** (11-15): metacarpal → tip
- **Ring** (16-20): metacarpal → tip
- **Little** (21-25): metacarpal → tip

## 3D Visualization

The 3D view displays:
- Red dots for joint positions (labeled 0-25)
- Green/blue lines connecting finger bones
- Updates at ~60 FPS

## Data Format

OSC data per joint: `X, Y, Z, qW, qX, qY, qZ`
- Position (X, Y, Z) and quaternion rotation (qW, qX, qY, qZ)

## APIs & Protocols

### StretchSense Open SDK

This application uses the **StretchSense Open SDK**, which is the official API for accessing real-time hand tracking data from StretchSense gloves. Key features:

- **Protocol**: Built on OSC (Open Sound Control)
- **Transport**: UDP packets
- **Default Ports**: 9000-9005
- **Data Rate**: ~60 Hz

The Open SDK is a lightweight, cross-platform solution that streams hand pose data without requiring game engine integration.

### OSC (Open Sound Control)

OSC is the underlying transport protocol used by the StretchSense Open SDK:

- **Message Address**: `/kinematic`
- **Payload Structure**: 5 header values + 26 joints × 7 floats
- **Per-Joint Data**: `[x, y, z, qw, qx, qy, qz]`
  - `x, y, z` — 3D position (meters)
  - `qw, qx, qy, qz` — Rotation quaternion

### Data Pipeline

```
StretchSense Gloves (Hardware)
        │
        ▼ Bluetooth/USB
XR Trainer Application
        │
        ▼ OSC/UDP (StretchSense Open SDK)
Python Code (this application)
        │
        ▼
GUI Display / 3D Visualization / CSV Export
```

### Python Libraries Used

| Library | Purpose |
|---------|---------|
| `python-osc` | Receives OSC messages from XR Trainer |
| `scipy` | Quaternion to Euler angle conversion |
| `numpy` | Numerical array operations |
| `matplotlib` | 3D visualization and animation |
| `tkinter` | GUI framework |

---

## Requirements

- Python 3.x
- numpy
- matplotlib
- python-osc
- scipy
- tkinter (included with Python)
