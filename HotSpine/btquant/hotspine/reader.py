import ctypes, time

lib = ctypes.CDLL("hotspine/build/libhotspine.so")

class Trade(ctypes.Structure):
    _fields_ = [
        ("ts_ns", ctypes.c_uint64),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
    ]

lib.hotspine_init.restype = ctypes.c_void_p
lib.hotspine_poll.argtypes = [ctypes.c_void_p, ctypes.POINTER(Trade)]

class HotSpineReader:
    def __init__(self):
        self.ring = lib.hotspine_init()
        self.trade = Trade()

    def poll(self):
        if lib.hotspine_poll(self.ring, ctypes.byref(self.trade)):
            return self.trade
        return None
