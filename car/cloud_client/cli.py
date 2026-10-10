"""Motor-free manual invocation and request preview."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .client import CloudAPIError, CloudClient
from .schema import SCHEMA_VERSION


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze road images with Qwen; never imports vehicle hardware")
    parser.add_argument("--image", action="append", required=True, help="JPEG/PNG/WebP; Realtime recovery 1-3 ordered frames, other contracts one; HTTP up to 8")
    parser.add_argument("--context", type=Path, help="JSON object with run ID, frame timestamps, local observations")
    parser.add_argument("--env-file", type=Path, help="Local environment file; default repository .env")
    parser.add_argument("--provider", choices=["qwen-realtime", "qwen", "openai-compatible"])
    parser.add_argument("--url", help="HTTP provider base URL or full chat/completions URL; Realtime uses --realtime-url")
    parser.add_argument("--model")
    parser.add_argument('--workspace-id',help='Beijing business workspace ID for Realtime')
    parser.add_argument('--realtime-url',help='Full WSS URL, no credentials in URL')
    parser.add_argument('--contract',choices=['road-observation-fast-v1','road-scene-v1','road-recovery-v1'])
    parser.add_argument('--timeout',type=float,help='Network request timeout, seconds; independent of 2s speed target')
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "xhigh"])
    parser.add_argument("--output", type=Path, help="New JSON result file; existing files are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview locally; no API key or network required")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required for live requests to preserve evidence")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; choose a new evidence file")
    try:
        context = json.loads(args.context.read_text(encoding="utf-8-sig")) if args.context else {}
        client = CloudClient(env_file=args.env_file, provider=args.provider, url=args.url,
                             model=args.model, reasoning_effort=args.reasoning_effort,
                             workspace_id=args.workspace_id,realtime_ws_url=args.realtime_url,contract=args.contract,timeout=args.timeout)
        if args.dry_run:
            payload=client.build_payload(args.image, context)
            record = {"mode": "dry-run", "network_called": False,
                      "provider": client.config.provider, "model": client.model,
                      "endpoint": payload.get('endpoint',client.url), "schema_version": client.config.contract,
                      'timeout_seconds':client.config.timeout,'hard_2s_deadline':False,
                      'http_generation_controls_applied':client.config.provider!='qwen-realtime',
                      "reasoning_effort": client.config.reasoning_effort if client.config.provider!='qwen-realtime' else None,
                      "max_tokens": client.config.max_tokens if client.config.provider!='qwen-realtime' else None,
                      "response_format": client.config.response_format if client.config.provider!='qwen-realtime' else None,
                      "images": client.last_image_manifest}
        else:
            if args.output.suffix.lower() != ".json":
                raise ValueError("result output must be a .json file")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Reserve the destination before spending on inference. Exclusive mode
            # protects evidence and catches permissions before the network call.
            with args.output.open("x", encoding="utf-8") as destination:
                try:
                    result = client.request_scene(args.image, context)
                except (ValueError, CloudAPIError, OSError) as exc:
                    json.dump({"mode": "failed-request", "error": str(exc),
                               "request": client.last_request_metadata}, destination, ensure_ascii=False, indent=2)
                    raise
                record = asdict(result)
                json.dump(record, destination, ensure_ascii=False, indent=2, allow_nan=False)
                destination.write("\n")
            # Do not dump the full raw response to the terminal.
            record = {"mode": "completed-request", "output": str(args.output.resolve()),
                      "request_id": result.request_id, "response_model": result.response_model,
                      "observation": result.scene,
                      "timings_ms": result.timings_ms, "usage": result.usage}
        print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, CloudAPIError, OSError) as exc:
        # Client error messages do not contain HTTP bodies or the API credential.
        print("Cloud scene request failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
