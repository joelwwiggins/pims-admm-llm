# Repository explainer video

**Output:** `pims_admm_llm_explainer.mp4` (~4–5 min narrated slide deck)

## Rebuild

```bash
cd ~/projects/pims-admm-llm
source .venv/bin/activate
pip install pillow imageio imageio-ffmpeg edge-tts   # if needed
python demos/video/make_explainer_video.py
```

Requires network for Microsoft Edge TTS voices (`en-US-GuyNeural`). Uses the `imageio-ffmpeg` bundled binary (no system `ffmpeg` required).

## Contents (10 scenes)

1. Title & value prop  
2. Planning problem (PIMS-style giant LP)  
3. Smart Refinery Planning Team  
4. ADMM coordination loop  
5. Why ADMM  
6. Live demo VERDICT numbers  
7. Shadow prices / make-buy-sell  
8. Repo layout  
9. How to run  
10. Proven + next steps  

## Artifacts

- `slides/*.png` — frame sources  
- `audio/*.mp3` — per-scene narration  
- `narration_full.mp3` — concatenated track  
- `build_report.json` — durations + path  
