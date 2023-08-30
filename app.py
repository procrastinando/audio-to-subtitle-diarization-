import whisperx
import gc
import pysrt
import gradio as gr
import torch

def device_change(device):
    if device.value == 'cpu':
        vram = gr.Checkbox.update(visible=False)
    else:
        vram = gr.Checkbox.update(visible=True)
    return vram

def diarization_check(diarization, auto_sp):
    if diarization:
        hf = gr.Textbox.update(visible=True)
        auto_sp = gr.Checkbox.update(value=True, visible=True)
        min_sp = gr.Slider.update(visible=False)
        max_sp = gr.Slider.update(visible=False)
    else:
        hf = gr.Textbox.update(visible=False)
        auto_sp = gr.Checkbox.update(value=True, visible=False)
        min_sp = gr.Slider.update(visible=True)
        max_sp = gr.Slider.update(visible=True)
    return hf, auto_sp, min_sp, max_sp

def auto_sp_change(auto_sp):
    if auto_sp == True:
        min_sp = gr.Slider.update(visible=False)
        max_sp = gr.Slider.update(visible=False)
    else:
        min_sp = gr.Slider.update(visible=True)
        max_sp = gr.Slider.update(visible=True)
    return min_sp, max_sp

def generate_srt(audio_file, model, device, vram, diarization, hf_token, auto_sp, min_sp, max_sp):
    if vram:
        compute_type = "int8"
        batch_size = 8
    else:
        batch_size = 16
        compute_type = "float16"

    if device=='cpu':
        compute_type = "int8"
    else:
        compute_type = "float16"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Generate subtitles
    if diarization:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        if auto_sp:
            diarize_segments = diarize_model(audio)
        else:
            if min_sp < max_sp:
                diarize_segments = diarize_model(audio, min_speakers=min_sp, max_speakers=max_sp)
            else:
                diarize_segments = diarize_model(audio, min_speakers=max_sp, max_speakers=max_sp)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        torch.cuda.empty_cache()

        speakers = set([r['speaker'] for r in result['segments']])
        sub_return = []
        for speaker in speakers:
            subs = pysrt.SubRipFile()
            for i in range(len(result['segments'])):
                if result['segments'][i]['speaker'] == speaker:
                    subs.append(pysrt.SubRipItem(i+1,start=pysrt.SubRipTime(seconds=result['segments'][i]['start']), end=pysrt.SubRipTime(seconds=result['segments'][i]['end']), text=result['segments'][i]['text']))
            subs.save(f'{speaker}.srt')
            sub_return.append(f'{speaker}.srt')
        return sub_return

    else:
        torch.cuda.empty_cache()
        subs = pysrt.SubRipFile()
        for i in range(len(result['segments'])):
            subs.append(pysrt.SubRipItem(i+1,start=pysrt.SubRipTime(seconds=result['segments'][i]['start']), end=pysrt.SubRipTime(seconds=result['segments'][i]['end']), text=result['segments'][i]['text']))
        subs.save('output.srt')
        return 'output.srt'



if torch.cuda.is_available():
    device_choices = ['cuda', 'cpu']
else:
    device_choices = ['cpu']

with gr.Blocks(title='ibarcena.net') as app:
    html = '''
        <a href='https://ibarcena.net/me'>
          <img src='https://ibarcena.net/content/images/2023/08/io2b-1.png alt='ibarcena.net/me'>
        </a>
    '''
    gr.HTML(html)

    audio = gr.Audio(source="upload", type='filepath', label="Audio File")
    model = gr.Dropdown(choices=['tiny', 'base', 'small', 'medium', 'large-v2'], value="small", label="Model Size")
    device = gr.Dropdown(choices=device_choices, value=device_choices[0], label="Device")

    if device.value == 'cpu':
        vram = gr.Checkbox(value=False, label="Low memory mode", visible=False)
        diarization = gr.Checkbox(default=False, label="Diarization", visible=False)
    else:
        vram = gr.Checkbox(value=False, label="Low memory mode", visible=True)
        diarization = gr.Checkbox(default=False, label="Diarization", visible=True)

    hf_token = gr.Textbox(label='Hugging Face Token', visible=False)
    with gr.Row():
        auto_sp = gr.Checkbox(label="Auto Number of speakers", value=True, visible=False)
        min_sp = gr.Slider(label="Min", value=1, minimum=1, maximum=10, step=1, visible=False)
        max_sp = gr.Slider(label="Max", value=1, minimum=1, maximum=10, step=1, visible=False)

    run = gr.Button('Run')
    output = gr.File(label="SRT File")

    device.change(fn=device_change, inputs=[device], outputs=[vram])
    diarization.change(fn=diarization_check, inputs=[diarization, auto_sp], outputs=[hf_token, auto_sp, min_sp, max_sp])
    auto_sp.change(fn=auto_sp_change, inputs=[auto_sp], outputs=[min_sp, max_sp])
    run.click(fn=generate_srt, inputs=[audio, model, device, vram, diarization, hf_token, auto_sp, min_sp, max_sp], outputs=[output])

    app.launch()