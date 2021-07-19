import subprocess
from multiprocessing import Process

# Services:
# 1. Start db (or not if sqlite?)
# 1. Start django server
# 2. Start embedding server
# 3. Serve react static files -- sanic?


def run_server():
    import uvicorn

    uvicorn.run("forager_server.asgi:application", host="0.0.0.0", port=8000)


def run_embedding_server():
    from forager_embedding_server import app

    app.app.run(host="0.0.0.0", port=5000)


def run_frontend():
    import uvicorn

    uvicorn.run("forager_frontend.app:app", host="0.0.0.0", port=4000)


class ForagerApp(object):
    web_server: Process
    embedding_server: Process
    file_server: Process

    def __init__(self):
        pass

    def _run_server(self, fn) -> Process:
        p = Process(target=fn, daemon=True)
        p.start()
        return p

    def run(self):
        web_server_wd = ""
        self.web_server = self._run_server(run_server)

        embedding_server_wd = ""
        self.embedding_server = self._run_server(run_embedding_server)

        file_server_wd = ""
        self.file_server = self._run_server(run_frontend)

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
