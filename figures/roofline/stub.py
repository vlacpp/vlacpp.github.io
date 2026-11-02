#!/usr/bin/env python3
"""Write a sparse stand-in for a remote GGUF: its real header, then zeros up to its real size.

    stub.py <url of a .gguf> <out.gguf>

Only the header is downloaded (HTTP range requests). The runtime loads the stand-in like the real
checkpoint and builds the same graphs, which is all the counter needs; the outputs are meaningless.
"""
import struct, sys, urllib.request

CHUNK = 4 << 20
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Remote:
    def __init__(self, url):
        self.url, self.buf, self.pos = url, b"", 0

    def fill(self, end):
        while len(self.buf) < end:
            start = len(self.buf)
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + CHUNK - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.buf += r.read()

    def read(self, n):
        self.fill(self.pos + n)
        self.pos += n
        return self.buf[self.pos - n:self.pos]

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def s(self):
        return self.read(self.u("Q")).decode("utf-8", "replace")

    def value(self, t):
        if t in SCALAR:
            return self.u(SCALAR[t])
        if t == 8:
            return self.s()
        if t == 9:
            et, n = self.u("I"), self.u("Q")
            for _ in range(n):
                self.value(et)
            return None
        raise ValueError(f"GGUF value type {t}")


def main(url, out):
    r = Remote(url)
    assert r.read(4) == b"GGUF", "not a GGUF"
    _, n_tensors, n_kv = r.u("I"), r.u("Q"), r.u("Q")
    align = 32
    for _ in range(n_kv):
        k, v = r.s(), r.value(r.u("I"))
        if k == "general.alignment":
            align = v
    for _ in range(n_tensors):
        r.s()
        for _ in range(r.u("I")):
            r.u("Q")
        r.u("I"), r.u("Q")
    data_start = (r.pos + align - 1) // align * align
    r.fill(data_start)
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    size = int(head.headers["Content-Length"])
    with open(out, "wb") as f:
        f.write(r.buf[:data_start])
        f.truncate(size)
    print(out, f"header {data_start} B, file {size} B")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
