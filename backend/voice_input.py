import speech_recognition as sr

def recognize_speech():
    """
    This function uses the SpeechRecognition library to capture audio from the microphone
    and convert it to text using Google's Web Speech API.
    """
    # Initialize recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("❗ Sorry, I couldn't understand that.")
        except sr.RequestError:
            print("❗ Could not request results from the API.")