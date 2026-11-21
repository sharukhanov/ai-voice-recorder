#!/usr/bin/env python3
"""BLE-разведка диктофона AIREC.

Использование:
    python3 tools/ble_scan.py                 # найти все BLE-устройства рядом
    python3 tools/ble_scan.py <ADDRESS>       # подключиться и выгрузить GATT-таблицу

На macOS ADDRESS — это UUID устройства (CoreBluetooth не отдаёт MAC-адреса).
Результат GATT-дампа сохраняется в captures/gatt_<имя>.txt
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from bleak import BleakClient, BleakScanner


async def scan(timeout: float = 10.0) -> None:
    print(f"Сканирую {timeout:.0f} с... (диктофон должен быть включён и НЕ подключён к телефону)\n")
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    rows = sorted(found.values(), key=lambda x: x[1].rssi, reverse=True)
    for dev, adv in rows:
        name = adv.local_name or dev.name or "?"
        print(f"{adv.rssi:>4} dBm  {dev.address}  {name}")
        if adv.service_uuids:
            print(f"           services: {', '.join(adv.service_uuids)}")
        if adv.manufacturer_data:
            for cid, data in adv.manufacturer_data.items():
                print(f"           mfr 0x{cid:04x}: {data.hex()}")
    print("\nДиктофон обычно ближе всех (самый сильный RSSI). Затем:")
    print("    python3 tools/ble_scan.py <ADDRESS>")


async def dump(address: str) -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    async with BleakClient(address, timeout=20.0) as client:
        out(f"# GATT dump {address}  {datetime.now().isoformat(timespec='seconds')}")
        for service in client.services:
            out(f"\n[Service] {service.uuid}  ({service.description})")
            for ch in service.characteristics:
                props = ",".join(ch.properties)
                out(f"  [Char] {ch.uuid}  handle={ch.handle}  props={props}  ({ch.description})")
                if "read" in ch.properties:
                    try:
                        val = await client.read_gatt_char(ch)
                        out(f"         value: {val.hex()}  {val!r}")
                    except Exception as e:  # noqa: BLE001
                        out(f"         read error: {e}")
                for d in ch.descriptors:
                    out(f"    [Desc] {d.uuid}  handle={d.handle}")

        # Короткое прослушивание notify-характеристик: вдруг устройство само что-то шлёт.
        notify_chars = [c for s in client.services for c in s.characteristics
                        if {"notify", "indicate"} & set(c.properties)]
        if notify_chars:
            out("\n# Подписка на notify на 10 с (можно понажимать кнопку на диктофоне)")

            def handler(ch, data: bytearray) -> None:
                out(f"  notify {ch.uuid}: {bytes(data).hex()}")

            for c in notify_chars:
                try:
                    await client.start_notify(c, handler)
                except Exception as e:  # noqa: BLE001
                    out(f"  start_notify {c.uuid} error: {e}")
            await asyncio.sleep(10)

    Path("captures").mkdir(exist_ok=True)
    path = Path("captures") / f"gatt_{address.replace(':', '')}.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    asyncio.run(dump(sys.argv[1]) if len(sys.argv) > 1 else scan())
