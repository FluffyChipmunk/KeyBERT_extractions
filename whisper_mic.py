import io
from pydub import AudioSegment
import speech_recognition as sr
import whisperx
import queue
import tempfile
import os
import threading
import click
import torch
import numpy as np
from keybert import KeyBERT
import time
from transformers.pipelines import pipeline


from main import KeyBERTextract, updateKeywordCounter

''''@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use", type=click.Choice(["cpu","cuda"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)'''


def whisper_mic(model, english,verbose, energy, pause,dynamic_energy,save_file,device):
    temp_dir = tempfile.mkdtemp() if save_file else None
    #there are no english models for large
    #if model != "large" and english:
        #model = model + ".en"
    audio_model = whisperx.load_model(model, device, compute_type="int8")
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    threading.Thread(target=record_audio,
                     args=(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir)).start()
    threading.Thread(target=transcribe_forever,
                     args=(audio_queue, result_queue, audio_model, english, verbose, save_file)).start()

    while True:

        #text = result_queue.get()['segments'][0]['text']
        #print(text)
        print(result_queue.get())



def record_audio(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir):
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        i = 0
        while True:
            #get and save audio to wav file
            audio = r.listen(source)
            if save_file:
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                filename = os.path.join(temp_dir, f"temp{i}.wav")
                audio_clip.export(filename, format="wav")
                print(filename)
                audio_data = filename
            else:
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_data = torch_audio

            audio_queue.put_nowait(audio_data)
            i += 1

keywordCounter = {


}



def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose, save_file):
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(audio_data,language='en')
        else:
            result = audio_model.transcribe(audio_data)

        if not verbose:
            predicted_text = result['segments'][0]['text']
            start = time.time()
            keyword = KeyBERTextract(predicted_text, kw_model, 1, 5, 3,keywordCounter)[0]
            end = time.time()
            print(end-start)
            updateKeywordCounter(keywordCounter, keyword)
            result_queue.put_nowait("You said: " + predicted_text + "\n" + "keyword: " + keyword)
        else:
            result_queue.put_nowait(result)

        if save_file:
            os.remove(audio_data)



kw_model = KeyBERT(model = "all-MiniLM-L6-v2")
#hf_model = pipeline("feature-extraction", model="phueb/BabyBERTa-1") #BERT model trained on CHILDES data
whisper_mic("tiny", True,False, 300, 0.8,False,True,"cpu")


'''if __name__ == "__main__":
    main()'''