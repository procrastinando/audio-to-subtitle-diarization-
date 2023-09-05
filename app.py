import whisperx
import gc
import os
import pysrt
import gradio as gr
import torch
from deep_translator import GoogleTranslator
from pytube import YouTube

def device_change(device):
    if device == 'cpu':
        vram = gr.Checkbox.update(value=False, visible=False)
        diarization = gr.Checkbox.update(value=False, visible=False)
    else:
        vram = gr.Checkbox.update(value=False, visible=True)
        diarization = gr.Checkbox.update(value=False, visible=True)
    return vram, diarization

def diarization_check(diarization, auto_sp):
    if diarization:
        hf = gr.Textbox.update(visible=True)
        auto_sp = gr.Checkbox.update(value=True, visible=True)
        min_sp = gr.Slider.update(visible=False)
        max_sp = gr.Slider.update(visible=False)
    else:
        hf = gr.Textbox.update(visible=False)
        auto_sp = gr.Checkbox.update(value=True, visible=False)
        min_sp = gr.Slider.update(visible=False)
        max_sp = gr.Slider.update(visible=False)
    return hf, auto_sp, min_sp, max_sp

def auto_sp_change(auto_sp):
    if auto_sp == True:
        min_sp = gr.Slider.update(visible=False)
        max_sp = gr.Slider.update(visible=False)
    else:
        min_sp = gr.Slider.update(visible=True)
        max_sp = gr.Slider.update(visible=True)
    return min_sp, max_sp

def generate_srt(yt, audio_file, model_size, device, vram, diarization, hf_token, auto_sp, min_sp, max_sp):
    # First make sure we have an audio file
    if yt != None or yt!= "":
        YouTube(yt).streams.filter(type='audio', subtype='mp4')[-1].download(filename='audio.mp4')
        audio_file = 'audio.mp4'

    global sub_return

    if vram:
        batch_size = 8
        compute_type = "int8"
    else:
        batch_size = 16
        compute_type = "float16"

    if device=='cpu':
        compute_type = "int8"
    else:
        compute_type = "float16"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
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
        if os.path.exists('output.srt'):
            os.remove('output.srt')
        subs.save('output.srt')
        sub_return = ['output.srt']
        return ['output.srt']

def translate_srt(output, translate):
    if sub_return == [] or output == None: # There is no output, neither a subtitle has been uploaded
        pass
    else:
        target_lang = language_dict[translate]
        sub2_return = []
        for i in sub_return:
            subs = pysrt.open(i)
            for sub in subs:
                sub.text = GoogleTranslator(source='auto', target=target_lang).translate(sub.text)
            file_name = i.split('.')[0] + f'-{target_lang}.srt'
            subs.save(file_name, encoding='utf-8')
            sub2_return.append(file_name)
        return sub2_return

##############################################################

language_dict = {
    "English": "en",
    "Bahasa (Indonesian)": "id",
    "Català (Catalan)": "ca",
    "Čeština (Czech)": "cs",
    "Dansk (Danish)": "da",
    "Deutsch (German)": "de",
    "Eesti (Estonian)": "et",
    "Español (Spanish)": "es",
    "Français (French)": "fr",
    "Italiano (Italian)": "it",
    "Latviešu (Latvian)": "lv",
    "Lietuvių (Lithuanian)": "lt",
    "Magyar (Hungarian)": "hu",
    "Nederlands (Dutch)": "nl",
    "Norsk (Norwegian)": "no",
    "Polski (Polish)": "pl",
    "Português (Portuguese)": "pt",
    "Română (Romanian)": "ro",
    "Slovenčina (Slovak)": "sk",
    "Suomi (Finnish)": "fi",
    "العربية (Arabic)": "ar",
    "Ελληνικά (Greek)": "el",
    "עברית (Hebrew)": "iw",
    "हिन्दी (Hindi)": "hi",
    "日本語 (Japanese)": "ja",
    "한국어 (Korean)": "ko",
    "Русский (Russian)": "ru",
    "中文 (简体) [Chinese Simplified]": "zh-CN",
    "中文 (繁體) [Chinese Traditional]": "zh-TW"
}

if torch.cuda.is_available():
    device_choices = ['cuda', 'cpu']
else:
    device_choices = ['cpu']
sub_return = []

with gr.Blocks(title='ibarcena.net') as app:
    html = '''
        <a href='https://ibarcena.net/me'>
            <img src='https://ibarcena.net/content/images/2023/08/io2b-1.png alt='ibarcena.net/me'>
        </a>
    '''
    gr.HTML(html)
    
    with gr.Row():
        with gr. Column():
            audio = gr.Audio(source="upload", type='filepath', label="Audio File")
            yt = gr.Textbox(label="Youtube link, leave empty to use an audio file")
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

        with gr.Column():
            output = gr.File(type='bytes', label="SRT File")
            translate = gr.Dropdown(list(language_dict.keys()), label="Translate to", value="English")
            translate_btn = gr.Button("Translate")
            output2 = gr.File(label="SRT File")

    device.change(fn=device_change, inputs=[device], outputs=[vram, diarization])
    diarization.change(fn=diarization_check, inputs=[diarization, auto_sp], outputs=[hf_token, auto_sp, min_sp, max_sp])
    auto_sp.change(fn=auto_sp_change, inputs=[auto_sp], outputs=[min_sp, max_sp])
    run.click(fn=generate_srt, inputs=[yt, audio, model, device, vram, diarization, hf_token, auto_sp, min_sp, max_sp], outputs=[output])
    translate_btn.click(fn=translate_srt, inputs=[output, translate], outputs=[output2])

    app.launch(share=False, debug=True)
