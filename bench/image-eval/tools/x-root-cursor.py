#!/usr/bin/env python3
# Gives the X server's root window the normal arrow cursor (what `xsetroot -cursor_name left_ptr`
# does). Under WSLg the root has none, and gpui never sets the arrow on its own windows (it assumes
# the arrow is already showing), so the pointer is invisible over the native app's plain areas.
import ctypes
x = ctypes.CDLL('libX11.so.6'); xc = ctypes.CDLL('libXcursor.so.1')
x.XOpenDisplay.restype = ctypes.c_void_p; x.XOpenDisplay.argtypes = [ctypes.c_char_p]
d = x.XOpenDisplay(None)
if d:
    x.XDefaultRootWindow.restype = ctypes.c_ulong; x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    xc.XcursorLibraryLoadCursor.restype = ctypes.c_ulong
    xc.XcursorLibraryLoadCursor.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    x.XDefineCursor.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
    x.XFlush.argtypes = [ctypes.c_void_p]; x.XCloseDisplay.argtypes = [ctypes.c_void_p]
    cursor = xc.XcursorLibraryLoadCursor(d, b'left_ptr')
    if cursor:
        x.XDefineCursor(d, x.XDefaultRootWindow(d), cursor)
    x.XFlush(d); x.XCloseDisplay(d)
