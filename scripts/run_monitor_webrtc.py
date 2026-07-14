#!/usr/bin/env python3
"""Run the queued UGREEN room-monitor WebRTC worker."""

from __future__ import annotations

import argparse
import asyncio
import signal

from posetestbot.monitoring.webrtc import run_monitor_webrtc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream the UGREEN room monitor over WebRTC.")
    parser.add_argument("monitor_root", help="Folder for the private worker status artifact.")
    parser.add_argument("--vendor-id", default="0c45")
    parser.add_argument("--product-id", default="2283")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:
            pass
    return await run_monitor_webrtc(
        args.monitor_root,
        stop_event=stop_event,
        vendor_id=args.vendor_id,
        product_id=args.product_id,
        width=max(1, args.width),
        height=max(1, args.height),
        fps=max(1, args.fps),
    )


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

