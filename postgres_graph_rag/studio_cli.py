"""Console entry point for the browser Demo Studio (`postgres-graph-rag-studio`).

Kept separate from `studio_app.py` so `create_app()` stays importable (and
testable via FastAPI's `TestClient`) without requiring `uvicorn` to actually
bind a port.
"""
import argparse
import os

# The Studio has no authentication and no CORS support -- anyone who can
# reach it can ingest documents, ask questions, and trigger paid provider
# calls. Binding beyond loopback is refused unless explicitly overridden.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="postgres-graph-rag-studio")
    parser.add_argument("--runtime-url", default=None, help="Defaults to $PGR_RUNTIME_URL.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--provider", choices=["offline", "openai", "gemini", "litellm"], default="offline",
        help="Fixed for the lifetime of this process -- every document ingested by any "
             "session uses this provider. Must match the dimension the target database's "
             "schema was provisioned with (see `postgres-graph-rag-demo setup --provider`).",
    )
    parser.add_argument(
        "--allow-non-loopback-bind", action="store_true",
        help="DANGEROUS: allows --host to be a non-loopback address (e.g. 0.0.0.0). The Studio "
             "has no authentication and no CORS protection -- anyone who can reach it can ingest "
             "documents, ask questions, and trigger paid provider calls. Only pass this if you "
             "understand and accept that risk (e.g. a firewalled, single-user remote sandbox).",
    )
    args = parser.parse_args()

    if args.host not in _LOOPBACK_HOSTS and not args.allow_non_loopback_bind:
        raise SystemExit(
            f"Refusing to bind to {args.host!r}: the Studio is a local, single-operator demo "
            "tool with no authentication or CORS protection. Use --host 127.0.0.1 (the default), "
            "or pass --allow-non-loopback-bind if you understand and accept the risk."
        )

    runtime_url = args.runtime_url or os.getenv("PGR_RUNTIME_URL")
    if not runtime_url:
        raise SystemExit(
            "PGR_RUNTIME_URL is required (env var or --runtime-url). "
            "Run `postgres-graph-rag-demo setup` first if you haven't already."
        )

    try:
        import uvicorn

        from . import playground_service
        from .studio_app import create_app
        from .playground_service import PlaygroundInputError
    except ImportError as exc:
        raise SystemExit(
            f"Missing a dependency required by the Studio ({exc}). "
            "Install it with: uv sync --extra studio"
        ) from exc

    try:
        litellm_kwargs = (
            playground_service.litellm_kwargs_from_env()
            if args.provider == "litellm"
            else {}
        )
        app = create_app(
            runtime_url=runtime_url,
            provider=args.provider,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            google_api_key=os.getenv("GOOGLE_API_KEY") or None,
            **litellm_kwargs,
        )
    except PlaygroundInputError as exc:
        raise SystemExit(f"Cannot start the Studio: {exc}") from exc

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
