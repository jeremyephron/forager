import argparse
import logging
import logging.config
import os
import os.path
import sys

from sanic import Sanic, response

from forager_frontend.log import LOGGING, init_logging

init_logging()

FILE_DIR = os.path.realpath(os.path.join(__file__, ".."))

app = Sanic(__name__, log_config=LOGGING)
app.static("/files", "/")
app.static("/", os.path.join(FILE_DIR, "build"))


@app.route("/")
async def index(request):
    return await response.file(os.path.join(FILE_DIR, "build/index.html"))


def main():
    app.run(host="0.0.0.0")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--port", type=str, default="4000")
    args.add_argument("--bind", type=str, default="0.0.0.0")
    main()
