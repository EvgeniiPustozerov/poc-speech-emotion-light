The novel mechanism to detect emotions from speech.
<br />
The demo - Telegram bot https://t.me/poc_speech_emotion_bot. The bot takes the recorded audio from the user or pre-loaded records (CREMA-D dataset), builds a spectrogram, preprocesses it, and extracts the voice features of four types (125 features in total):
<br />
• PRAAT features: main frequency, pitch, harmonic to noise, jitter, shimmer, intensity, formats;<br />
• MFSS features (librosa package): mean, variance, min, max, kurtosis, and skewness for 13 spectral ranges;<br />
• Pause features: number of pauses, spoken time, speech, and pause rate;<br />
• Non-linear model features (parselmouth package): mean, variance, min, max, kurtosis, and skewness for 16 core frequencies.<br />
The bot uses its features and a novel pre-trained xgboost model to detect the presence of the following six emotions in the speech: anger, disgust, fear, happiness, neutrality, and sadness. The bot gives information on whether there is a dominant emotion in speech and depicts the probability level of top 3 (in case of dominant emotion) or 2 (in case of neutral speech) emotions.<br />
<br />
Additional functionality implemented in the bot:<br />
• Automated speech-to-text recognition implemented with the state-of-the-art DistilBERT model (English) with the use of 4gram_big ARPA beam decoder to perform beam search and improve the quality of the text.<br />
• Punctuation and capitalization with a separate DistilBERT bidirectional transformer model.<br />
• Sentiment analysis (positive/negative text connotation) of the spoken text using a separate SST-2 BERT model for text classification.<br />
<br />
Experimental (TODO):<br />
• Speaker diarization: the bot separates the speakers and points out their emotions.