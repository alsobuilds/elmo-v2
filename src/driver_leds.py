#! /usr/bin/env python


"""

Driver node.

This node manages the neopixel matrix.

Uses the neopixel library to control the leds.

"""

import time
import board
import neopixel


import middleware as mw


class DriverLeds:
    """
    Hardware driver for the neopixel LED matrix.

    Reads LED color state from middleware and writes it to the physical
    neopixel hardware. Also handles fade animations requested by behaviours,
    keeping all hardware access within the driver layer.

    > ## Attributes

    ``node : mw.Node`` : Middleware node used for shutdown and logging.

    ``leds : mw.Leds`` : Middleware LED state shared with behaviours.

    ``colors : list[list[int]]`` : Local copy of the last written color state, used to detect changes.
    
    ``pixels : neopixel.NeoPixel`` : Neopixel hardware interface connected to GPIO pin D18.

    > ## Functions
    """

    def __init__(self):
        """
        Connect to middleware.
        Initialize node.
        Connect to neopixel.

        Initialises the middleware node and LED state, sets up a local
        color buffer to track the last written state, and opens a
        connection to the neopixel strip on GPIO pin D18 with
        ``auto_write`` disabled so frames are pushed explicitly.
        """
        self.node = mw.Node("driver_leds")
        self.leds = mw.Leds()
        self.colors = [[0, 0, 0]] * self.leds.number
        self.pixels = neopixel.NeoPixel(
            board.D18,
            self.leds.number,
            brightness=self.leds.brightness,
            auto_write=False,
        )
        print("brightness: %s, %s" % (self.leds.brightness, type(self.leds.brightness)))

    def _execute_fade(self):
        """
        Execute a hardware fade from current colors to leds.fade_target.

        Interpolates pixel colors from the current state to
        ``leds.fade_target`` across ``leds.fade_steps`` steps spread over
        ``leds.fade_duration`` seconds. Each interpolated frame is written
        directly to the neopixel hardware. The method aborts early if
        ``leds.fade_active`` is cleared externally between steps. On
        successful completion, the final target state is synced back to
        middleware and ``leds.fade_active`` is cleared.

        Parameters
        ----------
        None
            All inputs are read from the ``leds`` middleware object:
            ``leds.fade_target``, ``leds.fade_steps``, ``leds.fade_duration``,
            and ``leds.fade_active``.
        """
        start = [c[:] for c in self.colors]
        target = self.leds.fade_target[:]
        steps = max(1, self.leds.fade_steps)
        duration = max(0.0, self.leds.fade_duration)
        step_delay = duration / steps

        for i in range(steps + 1):
            # Check for external abort (e.g. blush behaviour cleared fade_active)
            if not self.leds.fade_active:
                return
            alpha = i / steps
            blended = [
                [
                    max(0, min(255, int(start[j][k] + (target[j][k] - start[j][k]) * alpha)))
                    for k in range(3)
                ]
                for j in range(self.leds.number)
            ]
            for idx in range(self.leds.number):
                self.pixels[idx] = blended[idx]
            self.pixels.show()
            self.colors = [c[:] for c in blended]
            time.sleep(step_delay)

        # Sync final state to middleware and clear the fade request
        self.leds.colors = [c[:] for c in target]
        self.colors = [c[:] for c in target]
        self.leds.fade_active = False

    def run(self):
        """
        Main loop.

        Marks the LED subsystem as ready in middleware, then enters a
        polling loop that runs until shutdown is requested. On each tick
        the loop checks for a pending fade request and delegates to
        ``_execute_fade`` if one is active. Otherwise it compares the
        middleware color state against the local buffer and, if a change
        is detected, clamps each channel to [0, 255], writes the new
        colors to the neopixel hardware, and updates the local buffer.
        On exit — whether from a ``KeyboardInterrupt`` or a middleware
        shutdown signal — all LEDs are turned off and the node is shut
        down cleanly.
        """
        try:
            self.leds.ready = True
            while not self.node.is_shutdown():
                time.sleep(0.1)
                # Handle fade request from behaviours
                if self.leds.fade_active:
                    self._execute_fade()
                    continue
                colors = self.leds.colors[:]
                if colors != self.colors:
                    # print("writing")
                    for i in range(self.leds.number):
                        r = max(0, min(255, int(colors[i][0])))
                        g = max(0, min(255, int(colors[i][1])))
                        b = max(0, min(255, int(colors[i][2])))
                        self.pixels[i] = [r, g, b]
                    self.pixels.show()
                    self.colors = colors
        except KeyboardInterrupt:
            pass
        finally:
            for i in range(self.leds.number):
                self.pixels[i] = [0, 0, 0]
            self.pixels.show()
            self.node.shutdown()


if __name__ == "__main__":
    driver = DriverLeds()
    driver.run()
