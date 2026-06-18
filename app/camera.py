# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V.
"""Camera acquisition – exclusively supports FLIR/PointGrey (Spinnaker SDK)."""
import struct
import threading
import time
import numpy as np

try:
    import PySpin
    PYSPIN_AVAILABLE = True
except ImportError:
    PYSPIN_AVAILABLE = False


class CameraError(Exception):
    pass


OPENRAMAN_CAL_IDENT = 0xCADA
OPENRAMAN_CAL_STRUCT = struct.Struct("<HBB4f")


def _checksum8(data: bytes) -> int:
    """OpenRAMAN checksum8: bitwise-not sum followed by bitwise-not."""
    total = 0
    for b in data:
        total = (total + ((~b) & 0xFF)) & 0xFF
    return (~total) & 0xFF


def list_cameras():
    """Return list of available FLIR cameras."""
    if not PYSPIN_AVAILABLE:
        return []
        
    available = []
    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            nodemap = cam.GetTLDeviceNodeMap()
            model_node = PySpin.CStringPtr(nodemap.GetNode("DeviceModelName"))
            serial_node = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
            name = f"FLIR {model_node.GetValue()} ({serial_node.GetValue()})"
            available.append(("flir", i, name))
            del cam
        cam_list.Clear()
        system.ReleaseInstance()
    except Exception as e:
        print(f"Error listing FLIR cameras: {e}")
        
    return available


class FlirCamera:
    """Wrapper for FLIR cameras using the Spinnaker SDK (PySpin)."""

    def __init__(self, index=0):
        if not PYSPIN_AVAILABLE:
            raise CameraError("PySpin (Spinnaker SDK) not installed. Please install the Spinnaker Python wrapper.")
        
        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()
        if self._cam_list.GetSize() <= index:
            raise CameraError(f"FLIR camera index {index} not found")
        
        self._cam = self._cam_list.GetByIndex(index)
        self._cam.Init()
        self._load_user_set2()
        
        # Set pixel format to Mono8 or Mono12/16 if possible
        nodemap = self._cam.GetNodeMap()
        pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
            for fmt in ["Mono16", "Mono12", "Mono8"]:
                entry = pixel_format.GetEntryByName(fmt)
                if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                    pixel_format.SetIntValue(entry.GetValue())
                    break
        self._set_bool_node("ReverseX", True)

        self._cam.BeginAcquisition()
        self._exposure = 0.1
        self._gain = 0.0
        self._roi_rows = None
        self._lock = threading.Lock()
        self.last_frame = None


    def _load_user_set2(self):
        nodemap = self._cam.GetNodeMap()
        selector = PySpin.CEnumerationPtr(nodemap.GetNode("UserSetSelector"))
        load = PySpin.CCommandPtr(nodemap.GetNode("UserSetLoad"))
        if not (PySpin.IsAvailable(selector) and PySpin.IsWritable(selector)):
            return
        if not (PySpin.IsAvailable(load) and PySpin.IsWritable(load)):
            return
        entry = selector.GetEntryByName("UserSet2")
        if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
            selector.SetIntValue(entry.GetValue())
            load.Execute()

    def _save_user_set2(self):
        nodemap = self._cam.GetNodeMap()
        selector = PySpin.CEnumerationPtr(nodemap.GetNode("UserSetSelector"))
        save = PySpin.CCommandPtr(nodemap.GetNode("UserSetSave"))
        if not (PySpin.IsAvailable(selector) and PySpin.IsWritable(selector)):
            return
        if not (PySpin.IsAvailable(save) and PySpin.IsWritable(save)):
            return
        entry = selector.GetEntryByName("UserSet2")
        if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
            selector.SetIntValue(entry.GetValue())
            save.Execute()

    @property
    def uid(self):
        nodemap = self._cam.GetTLDeviceNodeMap()
        serial = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber")).GetValue()
        return f"FLIR {serial}"

    @property
    def width(self):
        return self._cam.Width.GetValue()

    @property
    def height(self):
        return self._cam.Height.GetValue()

    def set_exposure(self, seconds):
        nodemap = self._cam.GetNodeMap()
        exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(exposure_auto) and PySpin.IsWritable(exposure_auto):
            exposure_auto.SetIntValue(exposure_auto.GetEntryByName("Off").GetValue())
        
        exposure_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsAvailable(exposure_time) and PySpin.IsWritable(exposure_time):
            # Spinnaker uses microseconds
            exposure_time.SetValue(seconds * 1_000_000)
            self._exposure = seconds

    def get_exposure(self):
        return self._exposure

    def set_gain(self, gain_db):
        nodemap = self._cam.GetNodeMap()
        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsAvailable(gain_auto) and PySpin.IsWritable(gain_auto):
            gain_auto.SetIntValue(gain_auto.GetEntryByName("Off").GetValue())
            
        gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsAvailable(gain_node) and PySpin.IsWritable(gain_node):
            gain_node.SetValue(gain_db)
            self._gain = gain_db

    def get_gain_db(self):
        return self._gain

    def set_roi(self, n_rows):
        self._roi_rows = n_rows

    def get_roi(self):
        return self._roi_rows if self._roi_rows else self.height

    def _set_bool_node(self, name, value):
        nodemap = self._cam.GetNodeMap()
        node = PySpin.CBooleanPtr(nodemap.GetNode(name))
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            node.SetValue(bool(value))

    def _user_value_nodes(self, write=False):
        nodemap = self._cam.GetNodeMap()
        selector = PySpin.CEnumerationPtr(nodemap.GetNode("UserDefinedValueSelector"))
        value = PySpin.CIntegerPtr(nodemap.GetNode("UserDefinedValue"))
        if not (PySpin.IsAvailable(selector) and PySpin.IsReadable(selector) and PySpin.IsWritable(selector)):
            raise CameraError("Camera does not expose UserDefinedValueSelector.")
        if not (PySpin.IsAvailable(value) and PySpin.IsReadable(value)):
            raise CameraError("Camera does not expose UserDefinedValue.")
        if write and not PySpin.IsWritable(value):
            raise CameraError("Camera UserDefinedValue is not writable.")
        return selector, value

    def _user_value_count(self, selector):
        entries = selector.GetEntries()
        if hasattr(entries, "GetSize"):
            iterable = [entries.GetByIndex(i) for i in range(entries.GetSize())]
        else:
            iterable = list(entries)
        count = 0
        for entry in iterable:
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                count += 1
        return count

    def _read_user_data(self, n_bytes):
        selector, value = self._user_value_nodes(write=False)
        count = self._user_value_count(selector)
        if n_bytes > count * 4:
            raise CameraError("Camera does not have enough user data slots.")

        out = bytearray()
        old = selector.GetIntValue()
        try:
            index = 0
            while len(out) < n_bytes:
                selector.SetIntValue(index)
                word = int(value.GetValue()) & 0xFFFFFFFF
                out.extend(struct.pack("<I", word))
                index += 1
        finally:
            selector.SetIntValue(old)
        return bytes(out[:n_bytes])

    def _write_user_data(self, data):
        selector, value = self._user_value_nodes(write=True)
        count = self._user_value_count(selector)
        if len(data) > count * 4:
            raise CameraError("Camera does not have enough user data slots.")

        old = selector.GetIntValue()
        try:
            for index in range((len(data) + 3) // 4):
                chunk = data[index * 4:(index + 1) * 4].ljust(4, b"\0")
                selector.SetIntValue(index)
                word = struct.unpack("<I", chunk)[0]
                if word >= 0x80000000:
                    word -= 0x100000000
                value.SetValue(word)
        finally:
            selector.SetIntValue(old)

    def _with_acquisition_paused(self, fn):
        with self._lock:
            was_streaming = self._cam.IsStreaming()
            if was_streaming:
                self._cam.EndAcquisition()
            try:
                return fn()
            finally:
                if was_streaming:
                    self._cam.BeginAcquisition()

    def get_calibration(self):
        """Read OpenRAMAN calibration coefficients from camera user memory."""
        def read():
            raw = bytearray(self._read_user_data(OPENRAMAN_CAL_STRUCT.size))
            ident, checksum, _reserved, *coeffs = OPENRAMAN_CAL_STRUCT.unpack(raw)
            if ident != OPENRAMAN_CAL_IDENT:
                raise CameraError("No OpenRAMAN calibration found in camera.")
            raw[2] = 0
            if checksum != _checksum8(raw):
                raise CameraError("Camera OpenRAMAN calibration checksum failed.")
            return [float(c) for c in coeffs]

        return self._with_acquisition_paused(read)

    def set_calibration(self, coeffs):
        """Write OpenRAMAN calibration coefficients to camera user memory."""
        def write():
            values = list(coeffs) + [0.0] * (4 - len(coeffs))
            raw = bytearray(OPENRAMAN_CAL_STRUCT.pack(
                OPENRAMAN_CAL_IDENT, 0, 0, *[float(c) for c in values[:4]]
            ))
            raw[2] = _checksum8(raw)
            self._write_user_data(bytes(raw))
            self._save_user_set2()

        self._with_acquisition_paused(write)

    def grab_frame(self):
        with self._lock:
            # Increase timeout for long exposures
            image_result = self._cam.GetNextImage(int(max(2000, (self._exposure * 1000) + 1000)))
            if image_result.IsIncomplete():
                image_result.Release()
                raise CameraError("Image incomplete")
            
            data = image_result.GetNDArray().astype(np.float64)
            image_result.Release()
            return data

    def acquire_spectrum(self):
        gray = self.grab_frame()
        self.last_frame = gray
        h, w = gray.shape
        
        if self._roi_rows and self._roi_rows < h:
            cy = h // 2
            r = self._roi_rows // 2
            roi = gray[max(0, cy - r): cy + r + 1, :]
        else:
            roi = gray

        signal = roi.sum(axis=0)
        saturation = gray.max(axis=0)
        roi_profile = gray.max(axis=1)

        return signal, saturation, roi_profile

    def set_hardware_roi(self, width=None, height=None, offset_x=0, offset_y=0):
        def adjust():
            nodemap = self._cam.GetNodeMap()
            
            # Offsets must be set to 0 first before changing Width/Height to avoid range violations
            ox_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
            if PySpin.IsAvailable(ox_node) and PySpin.IsWritable(ox_node):
                ox_node.SetValue(0)
            oy_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
            if PySpin.IsAvailable(oy_node) and PySpin.IsWritable(oy_node):
                oy_node.SetValue(0)

            # Width
            w_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            if PySpin.IsAvailable(w_node) and PySpin.IsWritable(w_node):
                w_max = w_node.GetMax()
                w_val = min(w_max, width) if width is not None else w_max
                inc = w_node.GetInc()
                if inc > 1:
                    w_val = (w_val // inc) * inc
                w_node.SetValue(w_val)
                
            # Height
            h_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            if PySpin.IsAvailable(h_node) and PySpin.IsWritable(h_node):
                h_max = h_node.GetMax()
                h_val = min(h_max, height) if height is not None else h_max
                inc = h_node.GetInc()
                if inc > 1:
                    h_val = (h_val // inc) * inc
                h_node.SetValue(h_val)
                
            # Now set the desired OffsetX and OffsetY
            if offset_x > 0 and PySpin.IsAvailable(ox_node) and PySpin.IsWritable(ox_node):
                ox_node.SetValue(min(ox_node.GetMax(), offset_x))
            if offset_y > 0 and PySpin.IsAvailable(oy_node) and PySpin.IsWritable(oy_node):
                oy_node.SetValue(min(oy_node.GetMax(), offset_y))
                
        self._with_acquisition_paused(adjust)

    def restore_hardware_roi(self):
        self._with_acquisition_paused(self._load_user_set2)

    def release(self):
        try:
            if hasattr(self, '_cam'):
                if self._cam.IsStreaming():
                    self._cam.EndAcquisition()
                if self._cam.IsInitialized():
                    self._cam.DeInit()
                # Remove reference to the camera before clearing the list
                del self._cam
                
            if hasattr(self, '_cam_list'):
                self._cam_list.Clear()
                del self._cam_list
                
            if hasattr(self, '_system'):
                if self._system.IsInUse():
                    pass # Keep system if others use it? No, ReleaseInstance is ref-counted.
                self._system.ReleaseInstance()
                del self._system
        except Exception:
            pass

    def __del__(self):
        self.release()


def Camera(type="flir", index=0):
    """Factory to create a FLIR camera object."""
    return FlirCamera(index)
