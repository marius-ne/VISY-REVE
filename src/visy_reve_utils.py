import numpy as np
import math

def _sig3_str(x: float) -> str:
    """Return a 3-digit string giving x to 3 significant figures, no decimal or exponent."""
    if x == 0:
        return "000"
    x = abs(x)
    e = math.floor(math.log10(x))
    # scale so that after rounding we have 3 total digits
    factor = 10**(2 - e)  # 2 because floor(log10) + 1 = digits left of decimal
    v = int(round(x * factor))
    s = f"{v:d}"
    # if rounding bumps it to 4 digits (e.g. 9.99→10.0→100), chop or pad:
    if len(s) > 3:
        s = s[:3]
    return s.zfill(3)

def encode_4x4(M: np.ndarray, to_numeric = False):
    """
    Encodes a 4x4 matrix into a single string.
    Args:
        M: Matrix.
        to_numeric: whether or not the output should be an integer or a string.
            If True (False by default) then minus signs and decimal points will
            be removed, so the number will be able to be reconstructed from the encoding.
    """
    assert M.shape == (4, 4)
    parts = []
    for i in range(4):
        for j in range(4):
            if to_numeric:
                parts.append(_sig3_str(M[i, j]))
            else:
                parts.append(str(M[i,j]))
    bigstr = "".join(parts)
    if to_numeric:
        return int(bigstr)
    else: 
        return bigstr