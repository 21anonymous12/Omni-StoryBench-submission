import whisper

def convert_speech_to_text(audio_file_path):

    model = whisper.load_model("large-v3")
    
    print(f"'{audio_file_path}' 변환을 시작합니다...")
    result = model.transcribe(audio_file_path)
    
    return result["text"]

