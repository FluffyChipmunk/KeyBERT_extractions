from whisper_mic import WhisperMic
import sys
from recStrategyImplementation import strategies, retrieval

mic = WhisperMic()
while True:
    result = mic.listen()
    recWord = strategies.KeyBERTextract(result)
    retrieved = retrieval.retrieveWords(recWord)
    print(result+" keyword: " + recWord + " retrieved: " + retrieved)

