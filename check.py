def _check_gst_python():
    from gi.repository import Gst
    try:
        Gst.Fraction(9001, 1)  # It's over NINE THOUSANDS!
    except TypeError:
        return False  # What, nine thousands?! There's no way that can be right
    return True

print(_check_gst_python())