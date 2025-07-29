import os
from moviepy.editor import *
from questions import *
from dataset import *
from voting_questions_normalization import *
import whisper
import shutil

def transcribe(path):
    model = whisper.load_model("turbo")
    model.to("cuda")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(path)
    chunksize = whisper.audio.N_SAMPLES
    nchunks = audio.shape[-1] // chunksize
    text = ""
    for x in range(nchunks+1):
        inp = whisper.pad_or_trim(audio[..., x*chunksize:(x+1)*chunksize])
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(inp, n_mels=128).to(model.device)

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        text += " " + result.text
        # print the recognized text
        print(result.text)
        del(result)
        del(mel)
    return text