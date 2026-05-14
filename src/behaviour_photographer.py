"""

Behaviour node.

When selected and triggered, it displays the camera on the screen,
centers the face.

Once the face is centered, the head stays on that position,
the countdown starts and, when it ends, it takes a picture and
sends it to be printed on an INSTAX.

"""

import os
import time
import subprocess
import cv2
import requests
import numpy as np
import threading
import middleware as mw


LOOP_RATE = 10
FRAME_W = 640
FRAME_H = 480
CONFIRM_FRAMES = 3


def take_picture(mjpeg_url):
    """
    Capture a single JPEG frame from an MJPEG stream and save it to disk.

    Reads the stream until a complete JPEG frame (SOI 0xFFD8 … EOI 0xFFD9) is
    assembled, decodes it, resizes to 480 px wide while preserving the aspect
    ratio, crops the height to 480 px, and writes the result to
    /tmp/captured_frame.png.

    Parameters
    ----------
    mjpeg_url : str
        Full URL of the MJPEG stream endpoint (e.g. 'http://localhost:8080/stream.mjpg').

    Returns
    -------
    None
    """
    # Send an HTTP GET request to the MJPEG stream URL
    response = requests.get(mjpeg_url, stream=True)
    if response.status_code == 200:
        stream = response.iter_content(chunk_size=1024)
        bytes = b''
        for chunk in stream:
            bytes += chunk
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.resize(frame, (480, int(frame.shape[0] / (frame.shape[1] / 480))))
                    frame = frame[0:480, :]
                    cv2.imwrite('/tmp/captured_frame.png', frame)
                    break
    else:
        print("Failed to retrieve MJPEG stream.")
    cv2.destroyAllWindows()


def print_picture(self):
    """
    Send the captured frame to the INSTAX printer via the instax_api CLI tool.

    Activates the instax virtual environment and invokes the print command as a
    subprocess. stdout, stderr, and the return code are all logged via the node
    logger for diagnosis.

    Parameters
    ----------
    self : BehaviourPhotographer
        The calling behaviour instance, used to access self.node for logging.

    Returns
    -------
    None
    """
    # Temporary solution since this tool needs other environment
    result = subprocess.run(
        "source /home/idmind/instax_api/instax/.venv/bin/activate && python -m instax.print -v 3 /tmp/captured_frame.png",
        shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    self.node.loginfo(f"[PHOTOGRAPHER] print_picture stdout: {result.stdout}")
    self.node.loginfo(f"[PHOTOGRAPHER] print_picture stderr: {result.stderr}")
    self.node.loginfo(f"[PHOTOGRAPHER] print_picture return code: {result.returncode}")


class BehaviourPhotographer:
    """
    Middleware behaviour that manages the full photo-taking and INSTAX printing flow.

    When the photographer mode is active and a sustained chest-touch is detected,
    the behaviour shows the camera feed on the onboard display, centres the
    detected face using pan/tilt servos, runs a 10-second LED countdown, captures
    a frame, and prints it on the connected INSTAX printer over Wi-Fi.

    > ## Attributes

    ``touch_sensors : mw.TouchSensors`` : Middleware touch sensor state used to detect chest touches.

    ``onboard : mw.Onboard`` : Middleware onboard display controller for images and the camera feed.
    
    ``camera : mw.Camera`` : Middleware camera state controller (URL, take_picture, taking_picture, error flags).

    ``behaviours : mw.Behaviours`` : Middleware behaviour configuration flags, used to read/write the photographer toggle.

    ``server : mw.Server`` : Middleware server helper for resolving icon resource URLs.

    ``leds : mw.Leds`` : Middleware LED controller used to display the countdown icons.

    ``pan : mw.Pan`` : Middleware pan servo controller for horizontal head movement.

    ``tilt : mw.Tilt`` : Middleware tilt servo controller for vertical head movement.

    ``printer : mw.Printer`` : Middleware printer state controller (wifi network name, connected flag).

    ``node : mw.Node`` : Middleware node used for shutdown and logging.

    ``detector : cv2.FaceDetectorYN`` : YuNet ONNX face detector configured for FRAME_W x FRAME_H input.

    ``latest_frame : numpy.ndarray or None`` : Most recent frame from the reader thread; None until stream is started.

    ``lock : threading.Lock`` : Mutex protecting access to latest_frame between the reader thread and main loop.

    ``stream : cv2.VideoCapture or None`` : MJPEG video capture; None when the stream is stopped.

    ``reader_running : bool`` : Flag used to signal the reader thread to stop when the stream is closed.

    ``smooth_cx : float`` : Exponentially smoothed horizontal face centre position (set on first track_face call).

    ``smooth_cy : float`` : Exponentially smoothed vertical face centre position (set on first track_face call).

    > ## Functions
    """

    def __init__(self):
        self.touch_sensors = mw.TouchSensors()
        self.onboard = mw.Onboard()
        self.camera = mw.Camera()
        self.behaviours = mw.Behaviours()
        self.server = mw.Server()
        self.leds = mw.Leds()
        self.pan = mw.Pan()
        self.tilt = mw.Tilt()
        self.printer = mw.Printer()
        self.node = mw.Node("behaviour_photographer")

        # face detection setup, same structure as behaviour_hello.py
        self.detector = cv2.FaceDetectorYN.create('/home/idmind/elmo-v2/src/yunet.onnx', '', (FRAME_W, FRAME_H))
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stream = None
        self.reader_running = False


    def start_stream(self):
        """
        Open the MJPEG stream and start the background reader thread.

        Creates a new VideoCapture connection to the local MJPEG endpoint,
        sets reader_running to True, and spawns a daemon reader thread.
        Sleeps 1 s to allow the first frames to arrive before returning.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stream = cv2.VideoCapture("http://localhost:8080/stream.mjpg")
        self.reader_running = True
        t = threading.Thread(target=self.reader, daemon=True)
        t.start()
        time.sleep(1.0)
        self.node.loginfo("Stream started.")

    def stop_stream(self):
        """
        Signal the reader thread to stop and release the MJPEG stream.

        Sets reader_running to False, waits 0.3 s for the thread to exit,
        releases the VideoCapture object, and clears latest_frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.reader_running = False
        time.sleep(0.3)
        if self.stream:
            self.stream.release()
            self.stream = None
        self.latest_frame = None
        self.node.loginfo("Stream stopped.")

    ####### 
    # these functions are the same of behaviour_hello.py 

    def reader(self):
        """
        Background thread target that continuously reads frames from the MJPEG stream
        and stores the latest one for use by the detection loop.

        Runs until reader_running is set to False. Failed reads are silently skipped.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        while self.reader_running:
            if self.stream:
                ret, frame = self.stream.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame

    def detect_face(self):
        """
        Grab the latest frame and run YuNet face detection on it.

        Only the highest-confidence (first) detected face is considered.
        Returns the pixel coordinates of its centre.

        Parameters
        ----------
        None

        Returns
        -------
        detected : bool
            True if at least one face was found in the latest frame, False otherwise.
        cx : float or None
            Horizontal pixel position of the face centre; None when no face is detected.
        cy : float or None
            Vertical pixel position of the face centre; None when no face is detected.
        """
        with self.lock:
            frame = self.latest_frame
        if frame is None:
            return False, None, None
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return False, None, None
        x, y, w, h = faces[0][:4]
        cx = x + w / 2
        cy = y + h / 2
        return True, cx, cy

    def track_face(self, cx, cy):
        """
        Update pan and tilt servo targets to keep the detected face centred in frame.

        Applies exponential smoothing (alpha=0.4) to the raw face position before
        computing the tracking error. A dead-band of ±8 % of frame width/height
        suppresses small jitter. The resulting angle adjustments are clamped to each
        servo's hardware limits before being written.

        Parameters
        ----------
        cx : float
            Horizontal pixel position of the face centre in the current frame.
        cy : float
            Vertical pixel position of the face centre in the current frame.

        Returns
        -------
        None
        """
        alpha = 0.4
        if not hasattr(self, 'smooth_cx'):
            self.smooth_cx = float(cx)
            self.smooth_cy = float(cy)
        self.smooth_cx = alpha * float(cx) + (1 - alpha) * self.smooth_cx
        self.smooth_cy = alpha * float(cy) + (1 - alpha) * self.smooth_cy

        error_x = (self.smooth_cx - FRAME_W / 2) / FRAME_W
        error_y = (self.smooth_cy - FRAME_H / 2) / FRAME_H

        if abs(error_x) < 0.08:
            error_x = 0
        if abs(error_y) < 0.08:
            error_y = 0

        pan_adjust = -error_x * 80
        tilt_adjust = error_y * 60

        new_pan = float(self.pan.current_angle) + pan_adjust
        new_tilt = float(self.tilt.current_angle) + tilt_adjust

        new_pan = max(self.pan.min_angle, min(self.pan.max_angle, new_pan))
        new_tilt = max(self.tilt.min_angle, min(self.tilt.max_angle, new_tilt))

        self.pan.angle = new_pan
        self.tilt.angle = new_tilt

    #######

    # all the center face function does is basically it tracks the face until its centered
    # and returns a boolean (true or false) to if its centered or not centered, respectivelly.

    def center_face(self):
        """
        Track the detected face with the servos until it is centred or a timeout elapses.

        Enables pan and tilt torque, then polls detect_face() at ~10 Hz for up to 5 s.
        On each frame where a face is found, track_face() is called and the normalised
        offset from the frame centre is checked. After CONFIRM_FRAMES consecutive frames
        within the ±8 % dead-band on both axes, the face is declared centred.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the face was successfully centred within the timeout, False otherwise.
        """
        self.node.loginfo("Centering face...")
        self.pan.enable = True
        self.tilt.enable = True
        # deadline of 5 seconds, so if the face isn't centered by then, give up
        deadline = time.time() + 5.0
        consecutive = 0 # Counts how many frames in a row the face has been "close enough" to center
        while time.time() < deadline:
            # grabs a frame and attempts to detect a face and returns its center coords if found
            detected, cx, cy = self.detect_face()
            if detected:
                self.track_face(cx, cy)
                error_x = (float(cx) - FRAME_W / 2) / FRAME_W # normalize the offset from frame center to a -0.5 … +0.5 range
                error_y = (float(cy) - FRAME_H / 2) / FRAME_H
                if abs(error_x) < 0.08 and abs(error_y) < 0.08: # face is within the ±8% "centered" zone on both axes
                    consecutive += 1
                    if consecutive >= CONFIRM_FRAMES: # only declare success after CONFIRM_FRAMES stable frames in a row
                        self.node.loginfo("Face centered.")
                        return True
                else:
                    # face drifted outside the zone, reset the stability counter
                    consecutive = 0
            time.sleep(0.1)
        # deadline elapsed without achieving a stable center lock
        self.node.loginfo("Timed out centering face.")
        return False


    def countdown(self):
        """
        Display a 10-to-0 countdown on the LED panel using numbered icon images.

        Saves the current LED colours, loads icons 10.png through 0.png one per
        second via the server URL resolver, then restores the original colours.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        old_colors = self.leds.colors
        for i in range(10, -1, -1):
            icon_name = "%d.png" % i
            url = self.server.url_for_icon(icon_name)
            self.leds.load_from_url(url)
            time.sleep(1.0)
        self.leds.colors = old_colors


    def disconnect_from_printer_wifi(self):
        """
        Disconnect from the INSTAX printer Wi-Fi network via nmcli.

        Reads the Wi-Fi network name from printer.wifi and issues a
        'nmcli con down' command. Updates printer.connected on success.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the disconnection command exited with code 0, False otherwise.
        """
        wifi = self.printer.wifi
        success = 0 == os.system("sudo nmcli con down id %s" % wifi)
        if success:
            self.printer.connected = False
            self.node.loginfo("Printer Disconnected")
        return success


    def connect_to_printer_wifi(self):
        """
        Connect to the INSTAX printer Wi-Fi network via nmcli.

        Reads the Wi-Fi network name from printer.wifi and issues a
        'nmcli con up' command. Updates printer.connected on success.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the connection command exited with code 0, False otherwise.
        """
        wifi = self.printer.wifi
        success = 0 == os.system("sudo nmcli con up id %s" % wifi)
        if success:
            self.printer.connected = True
            self.node.loginfo("Printer Connected")
        return success

    # this is also a new function, what it does is that it calls the functions in
    # a defined sequence:
    # show stream -> center face -> countdown -> take picture -> print -> restore

    def take_photo_flow(self):
        """
        Execute the full photo-taking and printing sequence in order.

        Steps performed:
        1. Start the MJPEG stream and display the camera feed on the onboard screen.
        2. Attempt to centre the detected face using pan/tilt servos (center_face).
        3. Run the 10-second LED countdown.
        4. Stop the stream so take_picture() can open a fresh connection.
        5. Capture a single frame to /tmp/captured_frame.png.
        6. Restore the onboard display (clear camera feed).
        7. Connect to the printer Wi-Fi, call print_picture(), then disconnect.
           On Wi-Fi connection failure, logs a warning and writes an error to camera.error.
        8. Clear the photographer flag to return to normal mode.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.node.loginfo("Photo flow started.")

        # start stream and show camera on screen
        self.start_stream()
        self.onboard.image = self.camera.url
        self.node.loginfo("Camera stream shown on screen.")
        time.sleep(1.0)

        # center face using servos
        self.center_face()

        # countdown
        self.countdown()

        # stop stream so take_picture can connect
        self.stop_stream()
        time.sleep(0.5)

        # take picture
        take_picture("http://localhost:8080/stream.mjpg")
        self.node.loginfo("Picture taken.")

        # hide camera from screen and restore normal mode
        self.onboard.image = None
        self.node.loginfo("Screen restored to normal.")

        # print
        if self.connect_to_printer_wifi():
            self.node.loginfo("Connected to printer wifi, printing picture.")
            print_picture(self)
            self.disconnect_from_printer_wifi()
        else:
            self.node.logwarn("Failed to connect to printer wifi.")
            self.camera.error = "failed to connect to printer wifi"

        # exit photographer mode
        self.behaviours.photographer = False
        self.node.loginfo("Photo flow finished. Returning to normal mode.")


    def run(self):
        """
        Main behaviour loop.

        Disconnects from the printer Wi-Fi on startup to avoid interfering with
        normal network traffic, then polls at LOOP_RATE Hz. On each tick:
        - Resets touch_counter and skips if photographer mode is not active.
        - Increments touch_counter while touch_sensors.touch_chest is True.
        - Once touch_counter reaches 10 (i.e. ~1 s of continuous touch), clears
          the counter, sets camera state flags, calls take_photo_flow(), and
          resets camera.taking_picture on completion.
        - Resets touch_counter if the chest touch is released before the threshold.

        Stops the stream and shuts down the middleware node on exit (including on
        KeyboardInterrupt or any other exception).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            self.node.loginfo("Starting behaviour.")
            touch_counter = 0
            self.disconnect_from_printer_wifi()

            while not self.node.is_shutdown():
                time.sleep(1.0 / LOOP_RATE)

                if not self.behaviours.photographer:
                    touch_counter = 0
                    continue

                # photographer mode is active, wait for sensor trigger
                if self.touch_sensors.touch_chest:
                    touch_counter += 1
                    if touch_counter >= 10:
                        touch_counter = 0
                        self.camera.take_picture = False
                        self.camera.taking_picture = True
                        self.camera.error = None
                        self.take_photo_flow()
                        self.camera.taking_picture = False
                else:
                    touch_counter = 0

        finally:
            self.node.loginfo("Shutting down.")
            self.stop_stream()
            self.node.shutdown()


if __name__ == '__main__':
    node = BehaviourPhotographer()
    node.run()