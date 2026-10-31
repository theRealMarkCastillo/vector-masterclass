"""Course entry points. Commands fail visibly instead of substituting recorded results."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def version():
    """Print the course package version."""
    from . import __version__

    print(__version__)


@app.command()
def bootstrap(model: str = "all"):
    """Explicitly download pinned model artifacts (or copy existing immutable snapshots)."""
    import json

    from .bootstrap import bootstrap_model
    from .course_audit import course_spec

    keys = (
        course_spec()["full_course_models"]
        if model == "full"
        else ["dense", "reranker", "generator", "sparse"]
        if model == "all"
        else [model]
    )
    for key in keys:
        result = bootstrap_model(key)
        print(json.dumps({"key": key, "bytes": result["bytes"], "source": result["source"]}))


@app.command()
def preflight():
    """Verify the kernel environment and actual Qdrant readiness."""
    import json

    from .preflight import preflight as check

    print(json.dumps(check(), indent=2))


@app.command()
def setup_service():
    """Explicitly create/reseed the local demo service data and identity simulation."""
    from .chunking import chunk_document
    from .config import root
    from .data import documents
    from .policy import PolicyStore, seed_demo_policy
    from .store import VectorIndex

    index = VectorIndex("vmc_service_v1")
    index.create()
    policy = PolicyStore(root() / "artifacts/service/policy.sqlite")
    try:
        docs = documents()
        for doc in docs:
            index.upsert(chunk_document(doc), document_id=doc.id)
        seed_demo_policy(policy, docs)
        print(
            "Local identity simulation seeded. Known tokens: demo-red-token, demo-blue-token, demo-public-token."
        )
        print(
            "This explicit setup restores demo grants; normal server startup preserves revocations."
        )
    finally:
        policy.close()
        index.close()


@app.command()
def serve(port: int = 8088):
    """Serve the teaching API on loopback only; run setup-service once first."""
    import uvicorn

    from .config import root
    from .policy import PolicyStore
    from .service import SearchService, create_app
    from .store import VectorIndex

    if not 1024 <= port <= 65535:
        raise typer.BadParameter("Use an unprivileged local port in [1024,65535]")
    path = root() / "artifacts/service/policy.sqlite"
    if not path.exists():
        raise typer.BadParameter("Run vmc setup-service before starting the API")
    policy = PolicyStore(path)
    index = VectorIndex("vmc_service_v1")
    index.verify_contract()
    service = SearchService(index, policy)
    try:
        uvicorn.run(create_app(service), host="127.0.0.1", port=port, access_log=False)
    finally:
        service.close()
        index.close()
        policy.close()
