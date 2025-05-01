from typing import Union, Dict, Any
import struct

try:
    import base58
    _b58decode = base58.b58decode
except ImportError:
    _ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    def _b58decode(txt: str) -> bytes:
        num = 0
        for c in txt:
            num = num * 58 + _ALPHABET.index(c)
        buf = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
        return b"\x00" * (len(txt) - len(txt.lstrip("1"))) + buf

# ── enum tag‑to‑name map ──────────────────────────────────────────────────────
_TAGS = {
    0: "Initialize",
    1: "Initialize2",
    2: "MonitorStep",
    3: "Deposit",
    4: "Withdraw",
    5: "MigrateToOpenBook",
}

_INSTR = {
    b"f\x06=\x12\x01\xda\xeb\xea": ("buy", [
        ("base_amount_out",      "<Q"),
        ("max_quote_amount_in",  "<Q"),
    ]),
    b"3\xe6\x85\xa4\x01\x7f\x83\xad": ("sell", [
        ("base_amount_in",       "<Q"),
        ("min_quote_amount_out", "<Q"),
    ]),
    b"\xf2#\xc6\x89R\xe1\xf2\xb6": ("deposit", [
        ("lp_token_amount_out",  "<Q"),
        ("max_base_amount_in",   "<Q"),
        ("max_quote_amount_in",  "<Q"),
    ]),
    b"\xb7\x12F\x9c\x94m\xa1\"":   ("withdraw", [
        ("lp_token_amount_in",   "<Q"),
        ("min_base_amount_out",  "<Q"),
        ("min_quote_amount_out", "<Q"),
    ]),
    b"\xe9\x92\xd1\x8e\xcfh@\xbc": ("create_pool", [
        ("index",                "<H"),
        ("base_amount_in",       "<Q"),
        ("quote_amount_in",      "<Q"),
    ]),
}

def _read_u8(buf: bytes, off: int):  return buf[off], off + 1
def _try_u64(buf: bytes, off: int):
    """Return (<value_or_None>, new_off).  No error if run out of bytes."""
    if off + 8 > len(buf):
        return None, off
    return struct.unpack_from("<Q", buf, off)[0], off + 8

def decode_raydium(data: Union[str, bytes]) -> Dict[str, Any]:
    """Decode Raydium AMM v4/v5 instruction data (safer version)."""
    raw: bytes = _b58decode(data) if isinstance(data, str) else bytes(data)
    tag, off   = _read_u8(raw, 0)
    out: Dict[str, Any] = {"tag": tag, "name": _TAGS.get(tag, "Unknown"), "fields": {}}

    # ---- tag‑specific parsing -----------------------------------------------
    if tag == 2:                            # MonitorStep (3×u16)
        for key in ("plan_order_limit", "place_order_limit", "cancel_order_limit"):
            if off + 2 <= len(raw):
                out["fields"][key], off = struct.unpack_from("<H", raw, off)[0], off + 2
            else:
                out["fields"][key] = None

    elif tag in (3, 4):                     # Deposit / Withdraw (1‑4×u64)
        keys = (
            ("max_coin_amount", "max_pc_amount", "base_side", "other_amount_min") if tag == 3
            else ("amount", "min_coin_amount", "min_pc_amount")
        )
        for k in keys:
            val, off = _try_u64(raw, off)
            if val is not None:
                out["fields"][k] = val

    elif tag == 0:                          # Initialize (u8 + u64)
        nonce, off       = _read_u8(raw, off)
        open_time, off   = _try_u64(raw, off)
        out["fields"]    = {"nonce": nonce, "open_time": open_time}

    elif tag == 1:                          # Initialize2 (u8 + 3×u64)
        nonce, off = _read_u8(raw, off)
        for k in ("open_time", "init_pc_amount", "init_coin_amount"):
            val, off = _try_u64(raw, off)
            out["fields"][k] = val
        out["fields"]["nonce"] = nonce

    # tag 5 (MigrateToOpenBook) carries no payload; unknown tags fall through
    return out

def decode_pumpfun(data: Union[str, bytes]) -> Dict[str, Any]:
    """
    Decode a Pump.fun AMM instruction `data` (base‑58 string or raw bytes).

    Returns
    -------
    dict   { 'name': <instruction_name_or_hex>, 'fields': {…} }
    """
    if not data:
        return None                                        # nothing to decode

    raw: bytes | None = None
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, str):
        try:
            raw = base58.b58decode(data)
        except Exception:
            # maybe it was base‑64
            if len(data) % 4 == 0:
                import base64
                try:
                    raw = base64.b64decode(data, validate=True)
                except Exception:
                    pass
    if raw is None:
        return {"name": "unknown_encoding", "raw": data}

    if len(raw) < 8:
        return {"name": "unknown_too_short", "raw": raw}

    disc, body = raw[:8], raw[8:]
    name, layout = _INSTR.get(disc, (disc.hex(), []))

    # Parse according to layout
    fields: Dict[str, Any] = {}
    offset = 0
    for fname, fmt in layout:
        size = struct.calcsize(fmt)
        if offset + size > len(body):
            break                                  # truncated → leave missing
        fields[fname] = struct.unpack_from(fmt, body, offset)[0]
        offset += size

    return {"name": name, "fields": fields}
