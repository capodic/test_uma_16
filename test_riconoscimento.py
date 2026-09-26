import sounddevice as sd

for i, dev in enumerate(sd.query_devices()):
    if "Linea (MCHStreamer Multi-" in dev["name"] or "miniDSP" in dev["name"] or "Linea (UMA16v2)" in dev["name"]:
        print(f"Device ID: {i} | Name: {dev['name']}")
        print(f"  Max Input Channels: {dev['max_input_channels']}")   # Deve restituire 16
        print(f"  Max Output Channels: {dev['max_output_channels']}") # Restituirà 2
    else:
        print(f"Device ID: {i} | Name: {dev['name']}")