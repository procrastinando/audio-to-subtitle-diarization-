# audio to subtitle (diarization)
This repository uses whisperX to generate subtitles using diarization
### Requirements:
* Python 3.10
* Anaconda

### 1. Install Pytorch
If CUDA:
A Hugging Face token is needed to use Diarization, make sure to accept the terms in this page: https://huggingface.co/pyannote/speaker-diarization
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If CPU:
Diarization will not be available
```
pip3 install torch torchvision torchaudio
```

### 2. Installation:
Clone the repository:
```
git clone https://github.com/procrastinando/audio-to-subtitle-diarization-
cd audio-to-subtitle-diarization-
```
Install requirements:
```
pip install -r requirements.txt
```

### 3. Run the webapp
```
python app.py
```
![image](https://github.com/procrastinando/audio-to-subtitle-diarization-/assets/74340724/a8087970-655f-4c15-9e80-05a75d4ee2a5)
