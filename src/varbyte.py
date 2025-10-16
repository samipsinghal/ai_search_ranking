"""
varbyte.py
-----------
Variable-byte encoding/decoding for non-negative integers.
Used to compress integer lists (docID deltas, term frequencies).
"""

__all__ = ["vb_encode_number", "vb_encode_list", "vb_decode_stream", "vb_decode_list"]

def vb_encode_number(n: int) -> bytes:
    """Encode a single non-negative integer using variable-byte coding."""
    if n < 0:
        raise ValueError("vb_encode_number expects non-negative integers")
    # Build the 7-bit chunks from least significant side, then mark last byte
    bytes_out = bytearray()
    while True:
        bytes_out.insert(0, n % 128)  # prepend 7 bits
        if n < 128:
            break
        n //= 128
    bytes_out[-1] |= 0x80            # mark final byte with high bit 1
    return bytes(bytes_out)

def vb_encode_list(numbers):
    """Encode an iterable of non-negative integers into a bytes blob."""
    out = bytearray()
    for n in numbers:
        out.extend(vb_encode_number(int(n)))
    return bytes(out)

def vb_decode_stream(data: bytes):
    """Generator that yields integers decoded from a varbyte-encoded bytes object."""
    n = 0
    for b in data:
        if b & 0x80:                 # last byte of this integer
            yield (n << 7) + (b & 0x7F)
            n = 0
        else:
            n = (n << 7) + b

def vb_decode_list(data: bytes):
    """Decode a varbyte-encoded bytes blob into a Python list of ints."""
    return list(vb_decode_stream(data))
