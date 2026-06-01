# Build an LLM-Powered Voice Assistant and Object Detection GUI for an Embodied AI Robot 

*How Claude, YOLO-style bounding boxes, the Web Speech API, and a live occupancy map came together into a complete robot control interface*

---

There is a particular joy in building something that looks like it belongs on the bridge of a starship but runs entirely in a browser tab.

CHIN-BOT is that thing. It is a real-time robot control GUI that combines a live LLM voice assistant (powered by Claude claude-sonnet-4-20250514), simulated YOLO-style object detection with bounding boxes, a D-pad locomotion controller, a live-updating occupancy minimap, and a full sensor telemetry dashboard — all in a single self-contained HTML file, no build step, no npm, no framework.

This article walks through every major design and engineering decision: why I made each choice, what the gotchas were, and what you can extend it into.

---

## The Problem: Embodied AI Needs a Control Interface

The phrase "embodied AI" gets used a lot in robotics research. At its core it means AI that is grounded in the physical world — a robot that perceives its environment through sensors, reasons about what it perceives, and acts on that reasoning through actuators.

Most embodied AI demos show a robot doing something impressive in a laboratory. What they rarely show is the *operator interface* — the GUI that a researcher, engineer, or remote operator uses to understand what the robot is thinking, correct its behaviour, and issue new commands.

That gap is what CHIN-BOT addresses. It is not the robot firmware. It is the human-facing control layer that makes the robot legible.

---

## Architecture at a Glance

The application is a three-column dashboard:

```
┌─────────────────┬────────────────────────────┬──────────────────┐
│  LEFT           │  CENTRE                    │  RIGHT           │
│                 │                            │                  │
│  Forward cam    │  Big vision viewport       │  Telemetry cards │
│  Telemetry      │  (object detection HUD)    │  Occupancy map   │
│  meters         │                            │  Object legend   │
│                 ├────────────────────────────│  Event log       │
│  D-pad          │  LLM chat panel            │                  │
│  controller     │  (voice + text input)      │                  │
│                 │                            │                  │
└─────────────────┴────────────────────────────┴──────────────────┘
```

Everything is drawn on `<canvas>` elements that animate at 60 fps via `requestAnimationFrame`. The LLM panel talks to the Anthropic API. The voice button uses the Web Speech API for both input (speech-to-text) and output (text-to-speech). No dependencies are loaded from CDNs — the only external resource is a Google Fonts import for `Share Tech Mono` and `Exo 2`.

---

## The Vision System: YOLO-Style Detection on Canvas

Real object detection requires a model — typically YOLOv8 running in ONNX.js, TensorFlow.js, or over a WebSocket to a Python backend. For this GUI the detection data is pre-seeded but structured exactly as a real YOLO pipeline would produce it: a list of objects, each with a class label, bounding box as normalised `[x, y, w, h]` fractions of the image, and a confidence score between 0 and 1.

```javascript
const SCENE_OBJECTS = [
  { cls:'PERSON', color:'#ff6b6b', bx:0.10, by:0.08, bw:0.18, bh:0.70, conf:0.94 },
  { cls:'CHAIR',  color:'#4ecdc4', bx:0.55, by:0.45, bw:0.20, bh:0.40, conf:0.87 },
  { cls:'TABLE',  color:'#45b7d1', bx:0.40, by:0.50, bw:0.35, bh:0.30, conf:0.91 },
  // ...
];
```

To render each detection, the canvas draw loop multiplies the normalised coordinates by the canvas dimensions at draw time. This means the bounding boxes scale correctly when the window is resized — a detail that often trips people up when hardcoding pixel values:

```javascript
function drawDetections(ctx, w, h) {
  SCENE_OBJECTS.forEach(obj => {
    const x = obj.bx * w,  y = obj.by * h;
    const bw = obj.bw * w, bh = obj.bh * h;

    // Bounding box stroke
    ctx.strokeStyle = obj.color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, bw, bh);

    // Corner accent brackets (the "targeting" look)
    const cs = 8;
    [[x,y,1,1],[x+bw,y,-1,1],[x,y+bh,1,-1],[x+bw,y+bh,-1,-1]]
      .forEach(([cx,cy,dx,dy]) => {
        ctx.beginPath();
        ctx.moveTo(cx + dx*cs, cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + dy*cs);
        ctx.stroke();
      });

    // Confidence label
    ctx.fillStyle = 'rgba(0,0,0,.75)';
    ctx.fillRect(x, y - 18, bw, 18);
    ctx.fillStyle = obj.color;
    ctx.fillText(`${obj.cls}  ${Math.round(obj.conf*100)}%`, x + 4, y - 5);
  });
}
```

The corner brackets — those small L-shapes at each corner of the bounding box — are the detail that makes the interface read as "robotic HUD" rather than "generic chart". Four two-line canvas paths, but they transform the visual impression entirely.

Objects are revealed progressively on boot (400ms delay between each), simulating the latency of a real inference pipeline initialising.

**To connect real detection:** Replace `SCENE_OBJECTS` with a WebSocket listener receiving JSON from a Python backend running `ultralytics` YOLOv8. The drawing code needs zero changes — it already consumes the same data structure YOLO produces.

---

## The LLM Voice Assistant: Giving the Robot a Voice

The most powerful part of the system is the LLM integration. Every message to Claude is prefixed with a system prompt that includes the robot's current state:

```javascript
function getSceneContext() {
  const visible = SCENE_OBJECTS.filter(o => detections.includes(o.cls));
  return `You are CHIN, an embodied AI robot assistant.
Current robot state:
- Position: (${Math.round(robotPos.x)}, ${Math.round(robotPos.y)}), heading ${Math.round(robotPos.heading)}°
- Speed: ${telemetry.speed.toFixed(1)} m/s, Battery: ${Math.round(telemetry.bat)}%
- Nearest obstacle: ${telemetry.dist.toFixed(1)}m

Detected objects (${visible.length} total):
${visible.map(o =>
  `  - ${o.cls} at (${Math.round(o.bx*100)}%, ${Math.round(o.by*100)}%) — ${Math.round(o.conf*100)}% confidence`
).join('\n')}

Respond concisely (2–4 sentences). Comment on what you see, plan navigation, or respond to commands.`;
}
```

This context injection is the key insight. Claude does not just answer generic questions — it answers *with knowledge of what the robot is currently perceiving*. Ask "is it safe to move forward?" and CHIN checks the ultrasonic distance reading. Ask "who is in the room?" and it describes the person at 94% confidence in the left quadrant.

The API call itself is a standard `fetch` to Anthropic's `/v1/messages` endpoint:

```javascript
const res = await fetch('https://api.anthropic.com/v1/messages', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': apiKey,
    'anthropic-version': '2023-06-01'
  },
  body: JSON.stringify({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 300,
    system: getSceneContext(),
    messages: [{ role: 'user', content: userText }]
  })
});
```

`max_tokens: 300` keeps responses crisp and robot-appropriate — a robot assistant should speak in short, precise sentences, not paragraphs.

### Voice Input and Output

Voice input uses the `SpeechRecognition` API (Chrome, Edge, Safari):

```javascript
recognition.onresult = e => {
  const text = e.results[0][0].transcript;
  document.getElementById('textInput').value = text;
  sendMessage(); // fires LLM call automatically
};
```

Voice output uses the `SpeechSynthesis` API — after every LLM response, the robot reads its reply aloud:

```javascript
const utt = new SpeechSynthesisUtterance(reply);
utt.rate = 1.05;
utt.pitch = 0.85; // slightly lower pitch sounds more robotic
speechSynthesis.speak(utt);
```

The combination — speak a command, hear the robot respond, see the detection overlay update — creates a genuinely immersive loop that feels far more capable than the individual pieces would suggest.

---

## Locomotion and the Occupancy Minimap

The D-pad controller is built from nine CSS grid cells. Only five are buttons; four are empty `<div>` spacers. Each button fires `mousedown` (start moving) and `mouseup` (stop) events:

```javascript
function moveRobot(dir) { moveDir = dir; }
function stopRobot()    { moveDir = null; }
```

Inside the animation loop, `moveDir` drives a simple dead-reckoning position update:

```javascript
if (moveDir === 'fwd')   { robotPos.y -= step; mapTrail.push({x: robotPos.x, y: robotPos.y}); }
if (moveDir === 'bwd')   { robotPos.y += step; }
if (moveDir === 'left')  { robotPos.x -= step; robotPos.heading -= 2; }
if (moveDir === 'right') { robotPos.x += step; robotPos.heading += 2; }
```

The trail array is capped at 80 points and drawn as a polyline on the minimap canvas. The robot icon is a small triangle rotated to `heading` degrees — rotated via `ctx.rotate()` inside a `ctx.save()` / `ctx.restore()` block so the rotation does not accumulate. Red rectangles at fixed positions simulate mapped obstacles.

This is not GPS navigation — it is a demonstration of how position state flows through the system. Replace the dead-reckoning with ROS topic data over a WebSocket and the minimap becomes a live odometry display.

---

## The Telemetry Dashboard

Four live-updating metric cards (speed, heading, distance, object count) plus a 5-channel ultrasonic sensor bar array simulate the kind of data a real mobile robot publishes continuously.

The values drift slightly on each animation frame to give the impression of live sensor noise:

```javascript
telemetry.cpu  = Math.min(95, Math.max(25, telemetry.cpu  + (Math.random() - .5) * 2));
telemetry.dist = Math.max(.5, Math.min(5,  telemetry.dist + (Math.random() - .5) * .1));
telemetry.bat  = Math.max(10, telemetry.bat - 0.0002); // slow drain
```

The battery meter turns from green to amber below 30% — a CSS class swap on the `.meter-fill` div, triggered by a threshold check in the update loop.

---

## The HUD Aesthetic

The visual language draws from real military and industrial HMI systems: dark backgrounds with low-saturation blue panels, monospace typography (`Share Tech Mono`), corner bracket targeting overlays, a scanline CSS pseudo-element across the whole viewport, and animated dot indicators for system status.

Two font families do all the work:
- `Share Tech Mono` for all data readouts, labels, and system identifiers — its slightly worn character spacing reads as genuinely machine-generated
- `Exo 2` for conversational text in the chat panel — geometric but readable at small sizes

The scanline effect is a single CSS rule:

```css
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent, transparent 2px,
    rgba(0,0,0,.08) 2px, rgba(0,0,0,.08) 4px
  );
  pointer-events: none;
  z-index: 9999;
}
```

Four lines of CSS. The dark horizontal bands are 2px of transparency followed by 2px of 8% black — subtle enough not to impair readability, visible enough to evoke a CRT display.

---

## What It Takes to Make This Real

The GUI is deliberately separated from the hardware layer. Everything that currently simulates a sensor value is a clear swap point for a real data source. Here is the mapping:

| GUI element | Simulated by | Real replacement |
|---|---|---|
| Object detection | Pre-seeded `SCENE_OBJECTS` array | YOLOv8 via WebSocket / ONNX.js |
| Camera feed | Canvas gradient scene | `getUserMedia()` or MJPEG stream |
| Position/heading | Dead-reckoning counter | ROS `/odom` topic via `rosbridge` |
| Ultrasonic array | Static meter bars | ROS `/scan` or `/sonar` topic |
| Battery / CPU | Slowly drifting values | Robot hardware API |
| LLM context | `getSceneContext()` string | Same function — already production-ready |

The LLM integration requires no changes at all when moving to real hardware. The context string already reads from the same variables that would be populated by real sensor data. That is the benefit of designing the state management carefully from the start.

---

## Running It

1. Download `EmbodiedAI_Robot_GUI.html`
2. Open it in Chrome or Edge (Firefox does not support `SpeechRecognition`)
3. Enter your Anthropic API key at the splash screen — or click **DEMO MODE** to explore without one
4. Click **📷 ANALYSE** for an instant LLM scene summary, or type / speak a command in the chat panel

The file is self-contained. No server required. No dependencies to install.

---

## What to Build Next

**Real camera feed.** Replace the canvas scene with a `<video>` element fed by `getUserMedia()` or a robot's MJPEG stream. Draw the detection overlay on a `<canvas>` layered over the video with `position: absolute`.

**ROS 2 integration.** `rosbridge_server` exposes a WebSocket interface to ROS topics. Subscribe to `/cmd_vel`, `/odom`, `/scan`, and `/camera/image_raw` and the GUI becomes a live robot control panel.

**Multi-robot support.** Add a robot selector to the header. Each robot maintains its own state object. The LLM context function already parameterises over robot state — pass in the selected robot's state and you get per-robot conversations for free.

**Tool calling.** Claude supports function/tool calling. Define tools like `navigate_to(x, y)`, `pick_up_object(class_name)`, and `report_status()`. The robot does not just discuss what to do — it does it, by returning structured tool calls that the GUI executes.

**Persistent memory.** Log each conversation turn with its associated detection state to `localStorage` or a backend. Let Claude summarise past sessions when the robot boots. Suddenly CHIN remembers that the bottle was on the table yesterday and it has moved.

---

## Closing Thoughts

The interesting thing about this project is that the hard part — the LLM integration — turned out to be the easiest part to build. Twenty lines of `fetch` and a well-crafted system prompt, and the robot gains a voice that understands its own sensor state.

The harder part was making the interface feel *robotic* — the corner brackets, the scanline overlay, the monospace telemetry, the blinking status dots. None of that is technically complex. All of it is worth doing carefully, because an interface that looks like it belongs to a serious machine gets taken seriously as a research tool.

CHIN-BOT is a starting point. The sensor data is simulated. The camera is painted. The map is approximate. But the architecture — scene context flowing into an LLM, LLM response spoken aloud, operator command processed back through the same pipeline — is exactly the loop that a real embodied AI robot control system would use.

The code is ready for real hardware. It is just waiting for the robot.

---

*The complete source — `EmbodiedAI_Robot_GUI.html` — is available on GitHub. If you are building robot interfaces, autonomous systems, or multimodal AI applications and found this useful, follow for more posts on applied AI engineering.*

---

**Tags:** Embodied AI · Robotics · LLM · Claude API · Object Detection · YOLO · JavaScript · Web Speech API · Robot Control · Human-Robot Interaction · Anthropic
