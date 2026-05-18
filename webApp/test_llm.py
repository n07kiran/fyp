import asyncio
from backend import what_it_is, WhatItIsRequest

async def main():
    req = WhatItIsRequest(
        predicted_class="Microcytic",
        cbc_data={"WBC": 8.49, "RBC": 4.37, "HGB": 10.1, "HCT": 29.9, "MCV": 68.4, "MCH": 23.1, "MCHC": 33.8, "PLT": 376.0, "MPV": 11.2, "RDW_CV": 14.6},
        shap_results=[{"feature": "MCV", "importance": 0.5}],
    )
    resp = await what_it_is(req)
    print("Response:", resp)

asyncio.run(main())
