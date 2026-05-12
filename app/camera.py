"""Camera acquisition – exclusively supports FLIR/PointGrey (Spinnaker SDK)."""
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
        
        # Set pixel format to Mono8 or Mono12/16 if possible
        nodemap = self._cam.GetNodeMap()
        pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
            for fmt in ["Mono12", "Mono16", "Mono8"]:
                entry = pixel_format.GetEntryByName(fmt)
                if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                    pixel_format.SetIntValue(entry.GetValue())
                    break

        self._cam.BeginAcquisition()
        self._exposure = 0.1
        self._gain = 0.0
        self._roi_rows = None
        self._lock = threading.Lock()

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
