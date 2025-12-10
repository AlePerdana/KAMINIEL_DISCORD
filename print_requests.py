import pyvts

v = pyvts.vts()
req = v.vts_request

print("=== AVAILABLE VTS REQUEST FUNCTIONS ===")
for name in dir(req):
    if name.startswith("request"):
        print("-", name)
print("========================================")
