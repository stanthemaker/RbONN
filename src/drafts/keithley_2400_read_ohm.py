"""Read resistance (ohms) from a Keithley 2400 SourceMeter over GPIB.

Runnable script built on PyVISA.  The 2400 measures ohms by sourcing a current
and measuring the voltage drop, so the output relay has to be ON for the reading
and is switched OFF again on exit (also on Ctrl-C / error).

Find the instrument first (ours answers at GPIB0::21; the 2400 factory default
is 24 -- override with ``--address``)::

    python src/drafts/keithley_2400_read_ohm.py --list

Single 2-wire reading at GPIB address 21::

    python src/drafts/keithley_2400_read_ohm.py

4-wire (Kelvin) reading, 10 samples 0.5 s apart, slower/quieter integration::

    python src/drafts/keithley_2400_read_ohm.py --four-wire --count 10 --interval 0.5 --nplc 10

Manual ohms mode -- you pick the test current (here 1 mA, 2 V compliance)
instead of letting the meter choose::

    python src/drafts/keithley_2400_read_ohm.py --current 1e-3 --compliance 2

Needs NI-VISA (or another GPIB-capable VISA) installed on this PC; the pure
Python ``@py`` backend cannot talk to a GPIB-USB adapter on Windows.
"""
from __future__ import annotations

import argparse
import sys
import time

# The 2400 returns +9.91e37 for an over-range / open-circuit reading.
OVERFLOW = 9.9e37


def open_2400(resource: str, timeout_s: float = 10.0, visa_library: str = ""):
    """Open the VISA session and confirm we are talking to a Keithley 24xx."""
    import pyvisa

    rm = pyvisa.ResourceManager(visa_library)
    inst = rm.open_resource(resource)
    inst.timeout = int(timeout_s * 1000)  # pyvisa timeout is in milliseconds
    inst.read_termination = "\n"
    inst.write_termination = "\n"

    idn = inst.query("*IDN?").strip()
    if "KEITHLEY" not in idn.upper() or "24" not in idn:
        print(f"warning: unexpected *IDN? response: {idn!r}", file=sys.stderr)
    return rm, inst, idn


def configure_ohms(
    inst,
    *,
    four_wire: bool = False,
    nplc: float = 1.0,
    current: float | None = None,
    compliance: float = 21.0,
) -> None:
    """Put the 2400 into resistance mode.

    ``current=None`` -> auto ohms (the meter picks the source current for the
    range); otherwise manual ohms sourcing ``current`` amps with ``compliance``
    volts as the voltage limit.
    """
    inst.write("*RST")
    inst.write("*CLS")
    inst.write(':SENS:FUNC "RES"')
    inst.write(":SENS:RES:RANG:AUTO ON")
    inst.write(f":SENS:RES:NPLC {nplc}")
    inst.write(f":SYST:RSEN {'ON' if four_wire else 'OFF'}")  # 4-wire remote sense

    if current is None:
        inst.write(":SENS:RES:MODE AUTO")
    else:
        inst.write(":SENS:RES:MODE MAN")
        inst.write(":SOUR:FUNC CURR")
        inst.write(f":SOUR:CURR {current}")
        inst.write(f":SENS:VOLT:PROT {compliance}")

    # Only return the resistance element so :READ? parses to a single float
    # (the default element list is VOLT,CURR,RES,TIME,STAT).
    inst.write(":FORM:ELEM RES")


def read_ohm(inst) -> float:
    """Trigger one measurement and return it in ohms (inf on over-range)."""
    value = float(inst.query(":READ?").strip())
    return float("inf") if value >= OVERFLOW else value


def fmt_ohm(value: float) -> str:
    if value == float("inf"):
        return "OVERFLOW (open?)"
    for scale, unit in ((1e6, "MOhm"), (1e3, "kOhm")):
        if abs(value) >= scale:
            return f"{value / scale:.6g} {unit}"
    return f"{value:.6g} Ohm"


def list_resources(visa_library: str = "") -> None:
    import pyvisa

    rm = pyvisa.ResourceManager(visa_library)
    names = rm.list_resources()
    if not names:
        print("no VISA resources found")
    for name in names:
        print(name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read ohms from a Keithley 2400 over GPIB.")
    p.add_argument("--list", action="store_true", help="list VISA resources and exit")
    p.add_argument("--address", type=int, default=21, help="GPIB primary address (default 21)")
    p.add_argument("--board", type=int, default=0, help="GPIB board index (default 0)")
    p.add_argument("--resource", help="explicit VISA resource string (overrides --address/--board)")
    p.add_argument("--visa", default="", help='VISA backend, e.g. "" for NI-VISA (default) or "@py"')
    p.add_argument("--four-wire", action="store_true", help="4-wire / Kelvin sensing (default 2-wire)")
    p.add_argument("--nplc", type=float, default=1.0, help="integration time in power-line cycles (0.01-10)")
    p.add_argument("--current", type=float, default=None,
                   help="manual ohms: source this many amps (default: auto ohms)")
    p.add_argument("--compliance", type=float, default=21.0,
                   help="manual ohms: voltage compliance limit in volts (default 21)")
    p.add_argument("--count", type=int, default=1, help="number of readings (default 1)")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
    p.add_argument("--timeout", type=float, default=10.0, help="VISA timeout in seconds")
    args = p.parse_args(argv)

    if args.list:
        list_resources(args.visa)
        return 0

    resource = args.resource or f"GPIB{args.board}::{args.address}::INSTR"
    rm, inst, idn = open_2400(resource, timeout_s=args.timeout, visa_library=args.visa)
    print(f"connected: {resource}  ->  {idn}")

    try:
        configure_ohms(
            inst,
            four_wire=args.four_wire,
            nplc=args.nplc,
            current=args.current,
            compliance=args.compliance,
        )
        inst.write(":OUTP ON")

        readings: list[float] = []
        for i in range(args.count):
            ohm = read_ohm(inst)
            readings.append(ohm)
            print(f"[{i + 1:>{len(str(args.count))}}/{args.count}] {fmt_ohm(ohm)}")
            if i + 1 < args.count:
                time.sleep(args.interval)

        finite = [r for r in readings if r != float("inf")]
        if len(finite) > 1:
            mean = sum(finite) / len(finite)
            std = (sum((r - mean) ** 2 for r in finite) / (len(finite) - 1)) ** 0.5
            print(f"mean {fmt_ohm(mean)}   std {fmt_ohm(std)}   n={len(finite)}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        # Never leave the source driving the DUT after the script exits.
        try:
            inst.write(":OUTP OFF")
        finally:
            inst.close()
            rm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
