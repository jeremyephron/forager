import os
import subprocess
import sys
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Tuple

# Services:
# 1. Start db (or not if sqlite?)
# 1. Start django server
# 2. Start embedding server
# 3. Serve react static files -- sanic?

LOG_DIR = Path("~/.forager/logs").expanduser()
os.environ.setdefault("FORAGER_LOG_DIR", str(LOG_DIR))


def run_server(q):
    # Run migrations if needed
    try:
        import django
        import uvicorn
        from django.core.management import call_command

        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "forager_server.settings")
        django.setup()
        call_command("makemigrations")
        call_command("migrate")

        print("Running django server...")
        q.put([True])
        uvicorn.run("forager_server.asgi:application", host="0.0.0.0", port=8000)
    finally:
        q.put([False])


def run_embedding_server(q):
    try:
        from forager_embedding_server import app

        print("Running embedding server...")
        q.put([True])
        app.app.run(host="0.0.0.0", port=5000)
    finally:
        q.put([False])


def run_frontend(q):
    try:
        import uvicorn

        print("Running frontend...")
        q.put([True])
        uvicorn.run("forager_frontend.app:app", host="0.0.0.0", port=4000)
    finally:
        q.put([False])


class ForagerApp(object):
    web_server: Process
    embedding_server: Process
    file_server: Process

    web_server_q: Queue
    embedding_server_q: Queue
    file_server_q: Queue

    def __init__(self):
        pass

    def _run_server(self, fn) -> Tuple[Process, Queue]:
        q = Queue()
        p = Process(target=fn, args=(q,), daemon=True)
        p.start()
        return p, q

    def run(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        web_server_wd = ""
        self.web_server, self.web_server_q = self._run_server(run_server)

        embedding_server_wd = ""
        self.embedding_server, self.embedding_server_q = self._run_server(
            run_embedding_server
        )

        file_server_wd = ""
        self.file_server, self.file_server_q = self._run_server(run_frontend)

        for name, q in [
            ("web", self.web_server_q),
            ("embedding", self.embedding_server_q),
            ("file", self.file_server_q),
        ]:
            started = q.get()
            if started:
                print(f"{name} server started")
            if not started:
                print(f"{name} server failed to start, aborting..")
                sys.exit(1)

        print("@@@ Ready to Forage! @@@")

    def join(self):
        self.web_server.join()
        self.embedding_server.join()
        self.file_server.join()


def main():
    app = ForagerApp()
    app.run()
    app.join()


if __name__ == "__main__":
    main()
