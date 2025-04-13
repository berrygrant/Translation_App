# translation.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def openai_translate(text, model="gpt-4o-mini-2024-07-18"):
    prompt = f"Translate the following Spanish text to English:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        translation = response.choices[0].message.content.strip()
        translation = translation.strip('"').strip("'")
        return translation
    except Exception as e:
        return f"Error in translation: {str(e)}"
