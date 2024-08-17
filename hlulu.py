from pydub import AudioSegment
from pydub.playback import play

# Load your TTS output
audio = AudioSegment.from_file("tts_output.wav")

# Example of altering pitch (raising pitch for a happier tone)
octaves = 0.5
new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
audio = audio.set_frame_rate(44100)

# Example of altering speed (slower for a more serious tone)
audio = audio.speedup(playback_speed=0.8)

# Play the audio
play(audio)
