# 📸 **RunReel — AI-Powered Running Detection & Vertical Reel Generator**

RunReel is a fully client-side React application that automatically detects **running actions** in images and videos using **TensorFlow.js + BlazePose**, extracts the relevant highlights, and generates polished **vertical HD reels (720×1280)** with optional background music, captions, and smooth zoom effects.

It is designed for creators, runners, sports analysts, and anyone who wants quick highlight reels from raw footage — without uploading media to a server.

---

## 🚀 **Key Features**

### 🧠 **AI Pose Detection (BlazePose + TensorFlow.js)**

* Detects whether a person in an image or video is running
* Computes a *running confidence* score
* Supports multiple file types (JPG, PNG, MP4, MOV)
* Processes entirely in the browser (no server required!)

### 🎬 **Vertical Reel Generator (720×1280, 30 FPS)**

* Person-centered smart cropping
* Subtle AI zoom-in / zoom-out effects for images
* Smooth frame rendering using `canvas.captureStream()`
* Handles videos frame-accurately using `requestVideoFrameCallback`

### 🎵 **Background Music Support**

* Upload any audio file (MP3, WAV, M4A, OGG)
* Perfect looping for full reel duration
* Smooth fade-in / fade-out (optional)
* Advanced audio pipeline using `AudioContext`, `GainNode`, and `MediaStreamDestination`

### 📝 **Custom Title & End Cards**

* Dynamic caption duration (based on number of words)
* Or fixed caption duration
* Auto-font sizing & clean layout

### 🌟 **Export**

* Downloads smooth HD vertical reel in **WebM (VP9 + Opus)**
* High quality: **6 Mbps video, 30 FPS**
* Works on Chrome, Edge, Opera, Firefox

---

## 🛠️ **Tech Stack**

| Component        | Technology                             |
| ---------------- | -------------------------------------- |
| Pose Detection   | TensorFlow.js + BlazePose              |
| UI               | React + Tailwind + lucide-react icons  |
| Video Processing | `<canvas>` + MediaRecorder API         |
| Audio Mixing     | Web Audio API (AudioContext, GainNode) |
| Effects          | Dynamic Cropping + Ken Burns zoom      |
| Output           | WebM (VP9/VP8 + Opus)                  |

---

# 📦 **Installation & Setup**

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/runreel.git
cd runreel
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start development server

```bash
npm start
```

Runs on **[http://localhost:3000/](http://localhost:3000/)**

---

# 🧩 **Project Structure Overview**

```
src/
 ├── RunReelApp.jsx     # Main application
 ├── index.js
 └── styles.css
public/
 └── index.html
```

Everything is contained inside `RunReelApp.jsx` — no backend or external server is used.

---

# ⚙️ **How the Pipeline Works**

Below is a simplified explanation for developers who want to create similar applications.

---

## **1. Pose Model Initialization**

```js
detectorRef.current = await poseDetection.createDetector(
    poseDetection.SupportedModels.BlazePose,
    { runtime: "tfjs", modelType: "lite" }
);
```

The BlazePose model runs entirely on WebGL in the browser.

---

## **2. Running Detection Logic**

Each pose frame is analyzed to compute:

* **Leg spread**
* **Hip width ratio**
* **Hip tilt**
* **Arm swing**
* **Stride factor**

Then a normalized running confidence value is computed:

```js
const runningConfidence = Math.min(
    1,
    (strideFactor * 0.6 + armSwing * 0.3 + hipTilt * 0.1) * 2.5
);
```

---

## **3. Smart Cropping & Person Centering**

Each frame centers automatically around:

* Nose position (preferred)
* Hip midpoint (fallback)

This prevents "jumping" crops and maintains the runner at the center of the vertical frame.

---

## **4. Frame-by-Frame Rendering**

A hidden 720×1280 `<canvas>` renders every frame:

```js
const videoStream = canvas.captureStream(FPS);
```

Images optionally receive a Ken Burns-style zoom effect:

* `zoomIn`
* `zoomOut`
* `none`

Videos use `requestVideoFrameCallback()` for smooth sync.

---

## **5. Audio Mixing (Music + Video)**

This uses the exact pipeline from the stable “A-version”:

```js
const ctx = new AudioContext();
const dest = ctx.createMediaStreamDestination();
const src = ctx.createBufferSource();
src.buffer = audioBuffer;
src.loop = true;

src.connect(gain).connect(dest);

mixedStream = new MediaStream([
  ...videoStream.getVideoTracks(),
  ...dest.stream.getAudioTracks(),
]);
```

This approach guarantees:

✔ Music plays until the very end
✔ No early cutoff
✔ Smooth fade-in/out
✔ Audio/video perfectly synced

---

## **6. Final Export via MediaRecorder**

```js
const recorder = new MediaRecorder(mixedStream, {
  mimeType: "video/webm;codecs=vp9,opus",
  videoBitsPerSecond: 6000000
});
```

Chunks are collected, then converted into a downloadable WebM file.

---

# 📘 **How to Build Similar Applications**

To replicate this system, you need to understand and combine:

### ✔ Pose detection → TensorFlow.js / WebGL

Use BlazePose or MoveNet.
BlazePose performs better for full-body running recognition.

### ✔ Video frame extraction → `<canvas>` & Web APIs

* Draw frames manually
* Apply cropping + transformations
* Capture at consistent FPS

### ✔ Music mixing → Web Audio API

* Decode audio (`decodeAudioData`)
* Loop or trim
* Combine with MediaStream
* Use GainNode for volume control

### ✔ Rendering final video → MediaRecorder

* Capture canvas
* Merge audio
* Export as WebM

This project demonstrates how all of these pieces fit together to build a fully client-side AI media engine.

---

# 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue to discuss what you’d like to improve.

---

# 📄 License

MIT License — free to use, modify, and build upon.
