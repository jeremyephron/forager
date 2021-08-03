import argparse
import logging
import os
import os.path

from sanic import Sanic, response

# Create a logger for the server
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Attach handlers
# Create a file handler for the log
if os.environ.get("FORAGER_LOG_DIR"):
    log_fh = logging.FileHandler(
        os.path.join(os.environ["FORAGER_LOG_DIR"], "embedding_server.log")
    )
    log_fh.setLevel(logging.DEBUG)
    log_fh.setFormatter(formatter)
    logger.addHandler(log_fh)


if os.environ.get("FORAGER_LOG_CONSOLE") == "1":
    # Create a console handler to print errors to console
    log_ch = logging.StreamHandler()
    log_ch.setLevel(logging.DEBUG)
    log_ch.setFormatter(formatter)
    logger.addHandler(log_ch)


FILE_DIR = os.path.realpath(os.path.join(__file__, ".."))

app = Sanic(__name__, log_config=None)
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
