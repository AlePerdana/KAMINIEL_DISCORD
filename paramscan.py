import asyncio
import pyvts
import os

PLUGIN_INFO = {
    "plugin_name": "ParamLister",
    "developer": "Kamitchi",
    "authentication_token_path": "./vts_param_token.txt",
}

async def main():
    v = pyvts.vts(plugin_info=PLUGIN_INFO)
    req = v.vts_request

    print("Connecting...")
    await v.connect()

    # authenticate [token file auto handled]
    try:
        await v.request_authenticate_token()
        await v.request_authenticate()
    except:
        # redo authentication
        if os.path.exists("./vts_param_token.txt"):
            os.remove("./vts_param_token.txt")
        print("Re-authorizing...")
        await v.request_authenticate_token()
        await v.request_authenticate()

    print("Connected.\nRequesting tracking parameter list...")

    # === GET ALL PARAMETER NAMES ===
    resp = await v.request(req.requestTrackingParameterList())
    params = resp.get("trackingParameters", [])

    param_names = [p["name"] for p in params]

    print("\n=== PARAMETER NAMES FOUND ===")
    for n in param_names:
        print(n)
    print("=============================\n")

    print("Now reading all parameter values...\n")

    # === LIVE LOOP ===
    try:
        while True:
            for n in param_names:
                val = await v.request(req.requestParameterValue(parameter=n))
                print(f"{n:25s} = {val.get('value')}")
            print("-" * 50)
            await asyncio.sleep(0.3)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        await v.close()


if __name__ == "__main__":
    asyncio.run(main())
