#!/usr/bin/env python3

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

C1_PRODUCT_ID = 0xFE0C
SERIAL_NUMBER_REQUEST = bytes((0x04, 0x00, 0x00, 0x30, 0x01))
SERIAL_RESPONSE_SIZE = 17
DEFAULT_API_BASE = "https://app.realplaybox.cn"
SCREEN_PARAM_PATH = "/jeecg-boot/openStage/device/queryScreenParam"


class ToolError(Exception):
    pass


def load_serial():
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as exc:
        raise ToolError(
            "pyserial is required. Install it with: python3 -m pip install pyserial"
        ) from exc
    return serial, list_ports


def available_ports():
    _, list_ports = load_serial()
    return sorted(list_ports.comports(), key=lambda port: port.device)


def select_port(requested_port):
    ports = available_ports()
    if requested_port:
        return requested_port

    matches = [port for port in ports if port.pid == C1_PRODUCT_ID]
    if len(matches) == 1:
        return matches[0].device
    if not matches:
        raise ToolError(
            "No C1 serial port with PID 0xFE0C was found. "
            "Use --list-ports to inspect devices or --port to select one explicitly."
        )
    raise ToolError(
        "Multiple C1 serial ports were found: "
        + ", ".join(port.device for port in matches)
        + ". Select one with --port."
    )


def read_serial_number(port_name, timeout):
    serial, _ = load_serial()
    try:
        with serial.Serial(
            port=port_name,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        ) as port:
            port.reset_input_buffer()
            written = port.write(SERIAL_NUMBER_REQUEST)
            port.flush()
            if written != len(SERIAL_NUMBER_REQUEST):
                raise ToolError(
                    f"Incomplete serial request: wrote {written} of "
                    f"{len(SERIAL_NUMBER_REQUEST)} bytes"
                )
            response = port.read(SERIAL_RESPONSE_SIZE)
    except (serial.SerialException, OSError) as exc:
        raise ToolError(f"Serial communication failed on {port_name}: {exc}") from exc

    if len(response) != SERIAL_RESPONSE_SIZE:
        raise ToolError(
            f"Expected {SERIAL_RESPONSE_SIZE} response bytes, received {len(response)}"
            + (f": {response.hex()}" if response else "")
        )
    return response.hex()[6:30]


def fetch_calibration(hardware_sn, api_base, token, language, timeout):
    url = api_base.rstrip("/") + SCREEN_PARAM_PATH + "?" + urllib.parse.urlencode(
        {"hardwareSN": hardware_sn}
    )
    headers = {
        "Accept": "application/json",
        "Accept-Language": language,
        "User-Agent": "c1-calibration-reader/1.0",
        "platform": "OPENSTAGE_AI",
        "prdCode": "OPENSTAGE_AI",
        "ver": "2.4.0",
    }
    if token:
        headers["X-Access-Token"] = token
        headers["X-Sign"] = token

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace")
        if exc.code == 401:
            raise ToolError(
                "Official API requires a valid login token. Set OPENSTAGEAI_TOKEN "
                f"or pass --token. Response: {detail}"
            ) from exc
        raise ToolError(f"Official API returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ToolError(f"Official API request failed: {exc}") from exc

    try:
        payload = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolError("Official API returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ToolError("Official API returned a non-object JSON response")
    if payload.get("success") is False:
        raise ToolError(
            f"Official API rejected the request: code={payload.get('code')}, "
            f"message={payload.get('message')}"
        )

    config = payload.get("result")
    if not isinstance(config, dict) or not config:
        raise ToolError(
            "Official API response does not contain a non-empty result object: "
            + json.dumps(payload, ensure_ascii=False)[:2000]
        )
    return config


def port_details():
    return [
        {
            "device": port.device,
            "description": port.description,
            "vid": f"0x{port.vid:04X}" if port.vid is not None else None,
            "pid": f"0x{port.pid:04X}" if port.pid is not None else None,
            "usb_serial_number": port.serial_number,
            "manufacturer": port.manufacturer,
        }
        for port in available_ports()
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read an OpenStage AI C1 hardware serial number over USB serial, "
            "then fetch its calibration from the official queryScreenParam API."
        )
    )
    parser.add_argument("--port", help="Serial port; auto-detects a single PID 0xFE0C device")
    parser.add_argument("--serial-timeout", type=float, default=3.0, help="Serial timeout in seconds")
    parser.add_argument("--api-timeout", type=float, default=30.0, help="API timeout in seconds")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Official API server base URL")
    parser.add_argument(
        "--token",
        default=os.environ.get("OPENSTAGEAI_TOKEN"),
        help="Login token; defaults to the OPENSTAGEAI_TOKEN environment variable",
    )
    parser.add_argument("--language", default="zh-CN", help="Accept-Language header")
    parser.add_argument("--output", type=Path, help="Also save the result to this JSON file")
    parser.add_argument(
        "--serial-only",
        action="store_true",
        help="Read and print only the hardware serial number; do not call the API",
    )
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.serial_timeout <= 0 or args.api_timeout <= 0:
        raise ToolError("Serial and API timeouts must be greater than zero")

    if args.list_ports:
        print(json.dumps(port_details(), ensure_ascii=False, indent=2))
        return 0

    port_name = select_port(args.port)
    hardware_sn = read_serial_number(port_name, args.serial_timeout)
    if args.serial_only:
        result = {"hardwareSN": hardware_sn}
    else:
        config = fetch_calibration(
            hardware_sn,
            args.api_base,
            args.token,
            args.language,
            args.api_timeout,
        )
        result = {"type": "device", "hardwareSN": hardware_sn, "config": config}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
