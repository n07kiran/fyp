import asyncio
from backend import what_it_is, WhatItIsRequest
from llm_provider import build_provider, LLMRequest

async def main():
    req = WhatItIsRequest(
        predicted_class="Microcytic",
        cbc_data={"WBC": 8.49, "RBC": 4.37, "HGB": 10.1, "HCT": 29.9, "MCV": 68.4, "MCH": 23.1, "MCHC": 33.8, "PLT": 376.0, "MPV": 11.2, "RDW_CV": 14.6},
        shap_results=[{"feature": "MCV", "importance": 0.5}],
    )
    
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
                "maxOutputTokens": 200,
            },
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers={"x-goog-api-key": key}, json=payload)
            print(resp.json())

asyncio.run(main())
