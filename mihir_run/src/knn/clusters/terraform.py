import asyncio
import json
from pathlib import Path
from shutil import copytree
import subprocess
from tempfile import TemporaryDirectory
import uuid


class TerraformModule:
    def __init__(self, module_path: Path, copy: bool = True):
        if copy:
            self.parent_dir = TemporaryDirectory()
            copytree(module_path, self.parent_dir.name)
            self.dir = Path(self.parent_dir) / module_path.name
        else:
            self.parent_dir = None
            self.dir = module_path

        self.id = str(uuid.uuid4())

        self._output = None
        self.ready = asyncio.Event()

    async def apply(self):
        proc = await asyncio.create_subprocess_exec("terraform", "init", cwd=self.dir)
        await proc.wait()

        proc = await asyncio.create_subprocess_exec(
            "terraform", "apply", "-auto-approve", cwd=self.dir
        )
        await proc.wait()

        self._output = None
        self.ready.set()

    @property
    def output(self):
        if self._output is None:
            proc = subprocess.run(
                ["terraform", "output", "-json"],
                stdout=subprocess.PIPE,
                cwd=self.dir,
            )
            self._output = {k: v["value"] for k, v in json.loads(proc.stdout).items()}
        return self._output

    async def destroy(self):
        self._output = None
        proc = await asyncio.create_subprocess_exec(
            "terraform", "destroy", "-auto-approve", cwd=self.dir
        )
        await proc.wait()

        if self.parent_dir:
            self.parent_dir.cleanup()
