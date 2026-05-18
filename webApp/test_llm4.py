import asyncio
from llm_provider import build_provider
from backend import what_it_is, WhatItIsRequest
import json

async def main():
    provider = build_provider()
    # we copy logic from llm_provider._call_gemini to print full response
    if hasattr(provider, '_rotator'):
        key = (await provider._rotator.next_key()).value
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{provider._model}:generateContent"
        import httpx
        system_prompt = (
            "You are an expert hematologist and medical AI explainer. "
            "Your task is to provide a concise, human-readable, and empathetic interpretation of what this predicted class means, "
            "why the model might have made this prediction based on the CBC and XAI results, and what the patient/doctor should look out for. \n"
            "Ensure your response is structured clearly with markdown, using bullet points for readability. \n"
            "Do NOT provide a definitive medical diagnosis. Keep it informative and helpful."
        )
        user_prompt = "Predicted Class: Microcytic"
        
        with open("/Users/kiran/Downloads/fyp/transformed_AneRBC_dataset/transformed_AneRBC-II/Microcytic/transformed_Original_images/0001_01_Microcytic.png", "rb") as f:
            img_bytes = f.read()
        import base64
        image_b64 = base64.b64encode(img_bytes).decode()
        
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}}
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 8192,
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers={"x-goog-api-key": key}, json=payload)
            print(json.dumps(resp.json(), indent=2))

asyncio.run(main())
