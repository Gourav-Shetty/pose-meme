# 🎭 PoseMeme

> **Real-time pose and hand gesture recognition that matches your movements to meme templates.**

Transform your body pose and hand gestures into hilarious meme matches in real-time! PoseMeme uses cutting-edge AI-powered vision technology to detect your movements and instantly display the perfect meme.

## ✨ Features

- **Real-time Pose Detection** — MediaPipe Tasks Vision tracks 33+ body keypoints at 30+ FPS with minimal latency
- **Hand Gesture Recognition** — Detects both hands with 21 keypoints each for precise gesture matching
- **Face Mesh Overlay** — Smooth, flicker-free 468-point face landmark visualization
- **Live Calibration** — Train custom pose-to-meme mappings with guided sample capture (50 samples per meme)
- **Intelligent Matching** — Combination of pose + hand similarity scoring (72% pose, 28% hands) with hysteresis thresholds
- **Default Display** — Shows your chosen image when idle, switches to matched meme when pose is clear enough
- **Session Persistence** — Temporary calibration samples that you can save or discard

## 📸 Demo

### Face & Pose Detection with Live Matching
![Pose Detection Demo](./Screenshot%202026-03-30%20161901.png)
*Real-time face mesh and pose skeleton with automatic meme matching.*

### Gesture Recognition in Action
![Gesture Demo](./Screenshot%202026-03-30%20161955.png)
*Hand gesture detection matching to the "peace" meme template.*

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- A webcam
- 8+ GB RAM (for smooth MediaPipe inference)

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/Gourav-Shetty/pose-meme.git
cd pose-meme

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at `http://localhost:5173` with hot module reloading.

## 🎮 How to Use

### 1. **Initial Training (One-time Setup)**
   - The app loads meme templates from `public/` folder
   - Extracts pose and hand landmarks from each image
   - Creates the baseline matching database

### 2. **Live Matching**
   - Simply pose in front of your webcam
   - The app analyzes your pose/hands in real-time
   - Displays the best matching meme on the right panel
   - Your default image appears when idle (no clear pose)

### 3. **Improve Matching with Calibration**
   - Click **"Calibrate Live"** to enter calibration mode
   - Hold each pose for 3–5 seconds per meme
   - System captures 50 diverse samples per pose automatically
   - Waits 3 seconds between poses so you can reset
   - Click **"Finish"** to apply or **"Cancel"** to discard
   - Click **"Clear Temp"** to reset temporary calibration

## ⚙️ Configuration

### Customize Meme List
Edit `metadata.json`:

```json
{
  "memeImages": ["1.jpg", "2.jpg", "peace.jpg"],
  "defaultImage": "default.jpg"
}
```

### Adjust Detection Parameters
Edit `src/App.tsx` constants:

```typescript
const FACE_STRIDE = 1;                    // Run face detection every frame
const FACE_MISS_TOLERANCE = 20;           // Persist face for 20 missed frames
const CALIBRATION_SAMPLES_PER_MEME = 50; // Samples per gesture during calibration
const CALIBRATION_MEME_DELAY_MS = 3000;  // Wait 3 seconds between memes
const ENTER_THRESHOLD = 0.5;              // Pose quality needed to match
const MINIMUM_MATCH_QUALITY = 0.45;       // Idle threshold to show default
```

## 📁 Project Structure

```
posememe/
├── src/
│   ├── App.tsx              # Main component with prediction loop
│   ├── lib/
│   │   └── matcher.ts       # Pose/hand vector matching logic
│   ├── main.tsx
│   └── index.css
├── public/
│   ├── default.jpg          # Shown when idle
│   ├── peace.jpg
│   ├── shock.jpg
│   └── ... (other meme images)
├── metadata.json            # Meme list configuration
├── package.json
├── tsconfig.json
├── vite.config.ts
└── index.html
```

## 🛠️ Build & Deploy

```bash
# Build for production
npm run build

# Output: dist/ folder with optimized bundle (~350KB gzipped)
```

Deploy the `dist/` folder to any static hosting (Vercel, Netlify, GitHub Pages, etc.).

## 🎯 How Matching Works

### Feature Extraction
1. **Pose Landmarks** — 33 keypoints normalized to bounding box (aspect-ratio preserved)
2. **Hand Vectors** — 21 keypoints × 2 hands, ordered left-to-right
3. **Flattened Vectors** — (x, y, visibility) triples concatenated into 1D arrays

### Similarity Scoring
- **Cosine Similarity** — Measures angular distance between feature vectors
- **Blended Score** — 72% pose bias + 28% hand bias
- **Hysteresis Thresholds** — Prevents flashing between memes:
  - `ENTER_THRESHOLD = 0.5` — Minimum score to show a meme
  - `EXIT_THRESHOLD = 0.4` — Score must drop below this to switch
  - `SWITCH_MARGIN = 0.05` — New meme must beat current by this much

### Calibration Strategy
- **Quality Gating** — Only accepts samples with 45%+ visibility
- **Diversity Filtering** — Rejects samples too similar to previous (cosine > 0.992)
- **Per-Gesture Buckets** — Accumulates 50 independent samples for each meme
- **Session Storage** — Keeps calibration in temporary state (not persisted to filesystem)

## 📊 Performance

- **Inference Speed** — 30+ FPS on modern CPUs (Intel i5+ / AMD Ryzen 5+)
- **Bundle Size** — 350 KB gzipped (includes all MediaPipe models)
- **Memory Footprint** — ~200–300 MB at runtime
- **Latency** — <100ms from webcam frame to meme display

## 🔑 Key Technologies

- **React 19** — UI framework with functional components & hooks
- **TypeScript 5.8** — Type safety across the codebase
- **MediaPipe Tasks Vision 0.10.34** — ML models:
  - PoseLandmarker
  - HandLandmarker
  - FaceLandmarker
- **Tailwind CSS 4.1** — Responsive styling
- **Vite 6.4** — Lightning-fast build tooling
- **react-webcam 7.2** — Webcam stream integration

## 🐛 Troubleshooting

### App Shows Wrong Meme
- Run calibration with clearer poses
- Ensure good lighting for accurate detection
- Check if pose is held long enough (3+ seconds per sample)

### Face Dots Blinking
- Already fixed! System now uses 20-frame persistence + temporal smoothing
- Normal flickering is reduced but not eliminated (due to detection variance)

### App Shows Default When Posing
- Pose might be too unclear — increase lighting
- Move closer to webcam for better keypoint visibility
- Quality threshold (`MINIMUM_MATCH_QUALITY`) might be too high

### Slow Performance
- Reduce resolution via browser DevTools
- Close other tabs consuming GPU
- Use Chrome/Edge (better hardware acceleration than Firefox)

## 🤝 Contributing

Got ideas? Found a bug? Feel free to **open an issue or submit a PR**!

## 📝 License

MIT License — feel free to use, modify, and distribute.

---

**Enjoy matching your moves to memes! 🎉**
