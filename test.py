import asyncio
import math
import time
import pyvts

PLUGIN_INFO = {
    "plugin_name": "LipSyncTest",
    "developer": "Kamitchi",
    "authentication_token_path": "./vts_token_test.txt",
}

async def main():
    vts = pyvts.vts(plugin_info=PLUGIN_INFO)

    await vts.connect()
    try:
        await vts.request_authenticate_token()
        await vts.request_authenticate()
    except:
        import os
        if os.path.exists("./vts_token_test.txt"):
            os.remove("./vts_token_test.txt")
        await vts.request_authenticate_token()
        await vts.request_authenticate()

    print("Connected! Sending sine wave (-1 to +1)...")

    start = time.time()

    while True:
        t = time.time() - start

        # Sine wave in full model range: -1.0 → +1.0
        val = math.sin(t * 3.0)

        print(f"MouthOpen = {val:.3f}")

        await vts.request(
            vts.vts_request.requestSetParameterValue(
                parameter="MouthOpen",
                value=float(val)
            )
        )

        await asyncio.sleep(0.03)

asyncio.run(main())
