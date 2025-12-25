#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from app.collector import Collector
from app.logging_utils import build_logger
from app.settings import load_config, load_settings
from app.storage import DataStorage


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings = load_settings(base_dir)
    logger = build_logger("data_manager", settings.log_dir)

    cfg = load_config(settings)
    storage = DataStorage(settings.db_path, logger)

    parser = argparse.ArgumentParser(prog="data-manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats")
    sub.add_parser("check")
    sub.add_parser("dedup")

    p_backfill = sub.add_parser("backfill")
    p_backfill.add_argument("days", type=int)

    p_run = sub.add_parser("run")
    p_run.add_argument("--once", action="store_true")

    p_web = sub.add_parser("web")
    p_web.add_argument("--host", type=str, default=settings.host)
    p_web.add_argument("--port", type=int, default=settings.port)

    args = parser.parse_args()

    collector = Collector(cfg, storage, logger)

    if args.cmd == "stats":
        print(storage.stats())
        return

    if args.cmd == "check":
        print(storage.check_integrity())
        return

    if args.cmd == "dedup":
        out = storage.remove_duplicates()
        if (out.get("price_removed", 0) + out.get("liquidity_removed", 0)) > 0:
            storage.optimize()
        print(out)
        return

    if args.cmd == "backfill":
        saved = collector.backfill(days=args.days)
        print({"saved": saved})
        return

    if args.cmd == "run":
        if args.once:
            collector.collect_once()
        else:
            collector.run_forever()
        return

    if args.cmd == "web":
        from app.web import create_app

        app = create_app(base_dir, settings, storage, logger)
        app.run(host=args.host, port=args.port, debug=False)
        return


if __name__ == "__main__":
    main()

