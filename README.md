# Forager
A rapid data exploration engine.

## Getting started

First, install Forager:
```bash
git clone forager-research/forager
pushd forager
pip3 install torch>=1.9 torchvision>=0.10.0 foragerpy
python3 -m build
pip3 install dist/*.whl
popd 
```

Now you can start up the Forager server by running:
```bash
forager-server
```

You can now access your Forager instance by typing [http://localhost:4000](http://localhost:4000) in your browser.

Note that the index page for Forager is empty. That's because we haven't loaded a dataset yet.

To do so, install the Python Forager client, foragerpy:

```bash
git clone forager-research/foragerpy
pushd foragerpy
poetry build
pip3 install dist/*.whl
popd
```

To load a dataset, you can start an asyncio-enabled REPL using `python3 -m asyncio` and then run the following:

```python
import foragerpy.client
client = foragerpy.client.Client(user_email="<YOUR@EMAIL.COM>")
await client.add_dataset('<DATASET_NAME>', '/path/to/train/images/directory, '/path/to/val/images/directory')
```

Now refresh the Forager web page and you should see your new dataset.
