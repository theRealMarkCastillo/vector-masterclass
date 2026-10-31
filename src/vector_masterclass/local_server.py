"""Context-managed loopback HTTP server for local integration/load exercises."""

import socket
import threading
import time
from contextlib import contextmanager

import uvicorn


@contextmanager
def local_server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not worker.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Local HTTP server failed to start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        worker.join(timeout=25)
        sock.close()
        if worker.is_alive():
            raise RuntimeError("Local HTTP server did not shut down within 25 seconds")
