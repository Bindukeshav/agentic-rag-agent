# TERA — Presentation Script (Concise)
### Matched to your actual deck. Short everywhere, deep on Objectives + Methodology/Flow.

---

## Slide 1 — Title
Good morning. My project is TERA — a Gesture-Controlled Explainable AI Assistant for Chest X-ray Analysis. I'm Amiya Ranjan Swain, with my team Ashish, Bikram, and Bindu Keshav, guided by Prof. Swetha H C.

## Slide 2 — Acknowledgement
Thank you to our Principal, HOD Dr. Nagashree N, coordinator Dr. Bhavana A, and our guide Prof. Swetha H C for their support.

## Slide 3 — Contents
I'll cover the problem, our objectives, research gap, methodology, architecture, roadmap, and scope.

## Slide 4 — Introduction
In sterile settings, doctors can't touch a keyboard or mouse. TERA lets them control chest X-ray analysis using only hand gestures — combining gesture recognition, lung/heart segmentation, explainable AI, and real-time visualization. Built entirely on public datasets, as a research prototype.

## Slide 5 — Problem Statement
Doctors can't touch devices during surgery, yet current viewers need a mouse and keyboard. AI predictions aren't explainable. No existing system combines gesture control with AI explanations. That gap is what we're solving.

---

## Slide 6 — Objectives (main focus — say this slowly, with weight)

We have five objectives, and I want to walk through *why* each one matters, not just list them.

**One — a contactless gesture interface.** This directly solves the sterility problem: no touching, no infection risk.

**Two — automatic lung and heart segmentation.** This is the foundation everything else stands on. We are not detecting TB or any disease in this project — I want to be completely clear about that. But no disease-detection AI can work on a raw X-ray; it first needs the lung region isolated from ribs, heart, and background. Segmentation is that essential first step. Get this boundary wrong, and nothing built on top of it — explanation or future detection — can be trusted. So this objective isn't a side task, it's the base layer the entire system depends on.

**Three — explainable AI outputs, using Grad-CAM and Attention Rollout.** The AI doesn't just give an answer — it shows *why*, so a doctor isn't trusting a black box.

**Four — improve doctor-AI interaction.** Gesture control isn't just navigation — it's how the doctor actively engages with the AI's reasoning, not passively glances at it.

**Five — a lightweight research prototype.** Runs on standard hardware, no special equipment — built to be demonstrated and evaluated academically, not deployed in a hospital today.

---

## Slide 7 — Research Gap
Existing systems segment images in isolation, lack gesture interaction, give limited AI explanation, and need expensive infrastructure. We integrate gesture recognition and explainable AI into one solution — that connection is our contribution. *(If pushed: "Remove segmentation, and three of our gestures lose their purpose — they exist to navigate the AI's explanation, not just move an image.")*

---

## Slide 8 — Methodology (main focus — walk the flow like a story)

*(Point to the diagram as you speak. Say it as one continuous flow, not five separate facts.)*

It starts with two inputs running in parallel. On one side, the webcam feed goes into MediaPipe for hand tracking, and the tracked landmarks get classified into a gesture. On the other side, the chest X-ray goes into our segmentation model, which produces the lung and heart boundary — and from that same output, we compute two explainability views: the Grad-CAM attention map, and the Attention Rollout view.

Both sides meet at one point: the gesture becomes the controller. Whatever the doctor's hand does decides which of those explanation views is shown on screen, right now, in real time.

That's the whole idea in one sentence: **the doctor's hand doesn't move an image — it moves through the AI's reasoning.**

---

## Slide 9 — System Architecture
*(Point to the diagram.)* This is that same flow, structurally: MediaPipe Hands feeds the gesture model, the X-ray feeds the segmentation model, both converge at a central controller that renders boundary, confidence, or attention based on the gesture — coordinated by our backend, which also logs the session.

## Slide 10 — Roadmap
*(Point to the diagram.)* Literature survey and dataset work done, gesture and segmentation modules built individually, now in integration and testing — which I'll show honestly on the progress slide.

## Slide 11 — Scope
TERA performs touchless lung/heart segmentation using Swin-UNETR, explainability via Grad-CAM and Attention Rollout, and gesture control via MediaPipe Hands and a GRU model — integrated through FastAPI and SQLite. This is a research prototype for education and decision *support* — not for clinical diagnosis or hospital deployment. Future scope: voice commands, cloud deployment, CT/MRI support, PACS integration, multi-user auth, mobile app.

## Slides 12–13 — References
Our foundation papers: U-Net and Swin-UNETR for segmentation, Grad-CAM for explainability, the transformer paper, and three public datasets — Montgomery, Shenzhen, JSRT.

## Slide 14 — Progress
Being honest: literature survey, dataset collection, and hand tracking are complete. Gesture recognition is ~70%, Swin-UNETR setup ~50%. Backend integration and full testing are still ahead. We'd rather show real progress than overstate it.

## Slide 15 — Thank You
Thank you — happy to take questions, and I can walk through the live demo if useful.

---

## Quick answers

| Asked | Say |
|---|---|
| "Detects TB?" | "No — only the lung/heart boundary. That's the necessary first step any future detector would need." |
| "Why segment if humans can see lungs?" | "The exact edge is fuzzy near the diaphragm and heart. The AI gives a precise, consistent, measurable boundary." |
| "What's actually new?" | "Gesture control and AI explainability both exist separately. Nobody connects gestures to *controlling explanations*. That's ours." |
| "Tested on patients?" | "No — public, de-identified datasets only." |
| "What's left?" | "Finishing gesture recognition and Swin-UNETR, then backend integration and testing." |

*Practice Slide 1 and Slide 15 twice tonight. Slides 6 and 8 are your strongest material — say them with confidence, they carry the whole project.*
