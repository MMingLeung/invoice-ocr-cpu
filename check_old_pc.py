#!/usr/bin/env python3
"""Probe OS / RAM / ISA / whether onnxruntime can be imported without SIGILL."""
from __future__ import annotations

import platform
import subprocess
import sys


def cpu_flags() -> set[str]:
    flags: set[str] = set()
    sysname = platform.system().lower()
    if sysname == "linux":
        try:
            text = open("/proc/cpuinfo", encoding="utf-8", errors="ignore").read().lower()
        except OSError:
            return flags
        for line in text.splitlines():
            if line.startswith("flags") or line.startswith("features"):
                flags.update(line.split(":", 1)[-1].split())
        return flags
    if sysname == "windows":
        ps = (
            "Get-WmiObject Win32_Processor | "
            "Select-Object -ExpandProperty Name"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=20,
            )
            name = (r.stdout or "").strip()
            if name:
                flags.add("cpu_name:" + name)
        except Exception as e:
            flags.add("wmi_error:" + type(e).__name__)
        # CPUID via a tiny python probe that cannot kill the parent
        return flags
    return flags


def probe_onnx() -> str:
    code = (
        "import onnxruntime as o; "
        "print(o.__version__); "
        "print(','.join(o.get_available_providers()))"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip().splitlines()
        tail = err[-1] if err else f"exit {r.returncode}"
        if r.returncode in (-4, 132, 3221225477, 3221225501):  # SIGILL / STATUS_ILLEGAL_INSTRUCTION
            return "SIGILL: " + tail
        return f"FAIL rc={r.returncode}: {tail}"
    return "OK " + " | ".join((r.stdout or "").strip().splitlines())


def ram_gb() -> str:
    try:
        import psutil  # type: ignore

        return f"{psutil.virtual_memory().total / (1024**3):.2f} GiB total, {psutil.virtual_memory().available / (1024**3):.2f} GiB available"
    except Exception:
        pass
    if platform.system().lower() == "linux":
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return f"{kb/1024/1024:.2f} GiB total (MemTotal)"
        except OSError:
            return "unknown"
    if platform.system().lower() == "windows":
        try:
            r = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            n = int((r.stdout or "0").strip() or "0")
            if n:
                return f"{n/1024**3:.2f} GiB total"
        except Exception:
            return "unknown"
    return "unknown"


def main() -> int:
    flags = cpu_flags()
    avx = "avx" in flags
    avx2 = "avx2" in flags
    print("os:", platform.platform())
    print("python:", sys.version.replace("\n", " "))
    print("ram:", ram_gb())
    if any(not f.startswith("cpu_name:") and not f.startswith("wmi_") for f in flags):
        print("avx:", avx, "avx2:", avx2)
    else:
        print("cpu:", ", ".join(sorted(flags)) or "(run Coreinfo if you need flags on Windows)")
        print("avx/avx2: on Windows, install Sysinternals Coreinfo or check CPU model (Sandy Bridge 2011 = AVX yes, AVX2 no)")
    print("onnxruntime:", probe_onnx())
    print("rule: SIGILL -> pin onnxruntime 1.16.3 then 1.11.1; Win7 -> Python 3.8.10 + ORT 1.11.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
