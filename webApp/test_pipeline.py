import asyncio
import httpx
import base64

async def main():
    async with httpx.AsyncClient(timeout=180.0) as client:
        # First get the images and cbc data by predicting
        with open("/Users/kiran/Downloads/fyp/transformed_AneRBC_dataset/transformed_AneRBC-II/Microcytic/transformed_Original_images/0001_01_Microcytic.png", "rb") as f:
            img_bytes = f.read()
        
        with open("/Users/kiran/Downloads/fyp/transformed_AneRBC_dataset/transformed_AneRBC-II/Microcytic/transformed_CBC_reports/0001_01_Microcytic.csv", "rb") as f:
            cbc_bytes = f.read()
            
        files = {
            "image": ("img.png", img_bytes, "image/png"),
            "cbc_csv": ("cbc.csv", cbc_bytes, "text/csv"),
        }
        resp = await client.post("http://localhost:8000/predict", files=files)
        res = resp.json()
        print("Predicted:", res["predicted_class"])
        
        # Now call what_it_is
        payload = {
            "predicted_class": res["predicted_class"],
            "cbc_data": res.get("cbc_data", {}),
            "shap_results": res.get("shap_results", []),
            "image_b64": res.get("original_image"),
            "gradcam_b64": res.get("gradcam_image")
        }
        
        llm_resp = await client.post("http://localhost:8000/what_it_is", json=payload, timeout=60.0)
        print("LLM Response:", llm_resp.json())

asyncio.run(main())
