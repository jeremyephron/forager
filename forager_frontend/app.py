import argparse
import logging
import logging.config
import os
import os.path
import sys

from sanic import Sanic, request, response
from sanic.exceptions import NotFound

from forager_frontend.log import LOGGING, init_logging

init_logging()

FILE_DIR = os.path.realpath(os.path.join(__file__, ".."))

app = Sanic(__name__, log_config=LOGGING)
app.static("/files", "/")
app.static("/static", os.path.join(FILE_DIR, "build", "static"))
app.static("/manifest.json", os.path.join(FILE_DIR, "build", "manifest.json"))
app.static("/favicon.ico", os.path.join(FILE_DIR, "build", "favicon.ico"))
app.static("/robots.txt", os.path.join(FILE_DIR, "build", "robotx.txt"))
app.static("/logo192.png", os.path.join(FILE_DIR, "build", "logo192.png"))
app.static("/logo512.png", os.path.join(FILE_DIR, "build", "logo512.png"))


@app.route("/")
@app.exception(NotFound)
async def index(request, param=""):
    return await response.file(os.path.join(FILE_DIR, "build/index.html"))


def main():
    app.run(host="0.0.0.0")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--port", type=str, default="4000")
    args.add_argument("--bind", type=str, default="0.0.0.0")
    main()
