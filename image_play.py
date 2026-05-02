import sys
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def main():
    # FPS configuration
    fps = 30
    
    # Path configuration
    path02 = "/home/jinbeom/workspace/2013_05_28_drive_0004_sync/image_02/data_rgb/image_%06d.png"
    path03 = "/home/jinbeom/workspace/2013_05_28_drive_0004_sync/image_03/data_rgb/image_%06d.png"

    # Initialize GStreamer
    Gst.init(None)

    # Pipeline definition with dynamic FPS
    pipeline_str = (
        f"nvstreammux name=m batch-size=2 width=1400 height=1400 ! "
        f"nvmultistreamtiler rows=1 columns=2 width=2800 height=1400 ! "
        f"nvvideoconvert ! nveglglessink "
        f"multifilesrc location=\"{path02}\" caps=\"image/png,framerate={fps}/1\" ! "
        f"pngdec ! videorate ! video/x-raw,framerate={fps}/1 ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_0 "
        f"multifilesrc location=\"{path03}\" caps=\"image/png,framerate={fps}/1\" ! "
        f"pngdec ! videorate ! video/x-raw,framerate={fps}/1 ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_1"
    )

    print(f"Starting pipeline with FPS: {fps}")
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return

    # Start playing
    pipeline.set_state(Gst.State.PLAYING)

    # Create a GLib MainLoop to handle events
    loop = GLib.MainLoop()

    # Bus for message handling
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of Stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err.message}")
            loop.quit()
        return True

    bus.connect("message", on_message, loop)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
