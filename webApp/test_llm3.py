import asyncio
from llm_provider import build_provider
from backend import what_it_is, WhatItIsRequest

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
        user_prompt = "Predicted Class: Microcytic\nCBC Data:\n- WBC: 8.49\n- RBC: 4.37\n- HGB: 10.1\n- HCT: 29.9\n- MCV: 68.4\n- MCH: 23.1\n- MCHC: 33.8\n- PLT: 376.0\n- MPV: 11.2\n- RDW_CV: 14.6\n\nSHAP Analysis (Top 5 Features):\n- MCV: 0.5\n- HCT: 0.2\n- HGB: 0.1\n- MCH: 0.1\n- MCHC: 0.1\n\nPlease provide the 'What it is?' interpretation."
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1000,
            },
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers={"x-goog-api-key": key}, json=payload)
            print(resp.json())

asyncio.run(main())
