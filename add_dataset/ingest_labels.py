from collections import defaultdict
from email.utils import parseaddr
import os
import json
from tqdm import tqdm

import aiohttp
import click

from utils import make_identifier, unasync


SERVER_URL = os.environ["SERVER_URL"]
ADD_ANNOTATIONS_ENDPOINT = "api/add_annotations_by_internal_identifiers_v2"


def parse_label(label):
    if isinstance(label, str):
        return (label, "POSITIVE")
    elif isinstance(label, dict) and "category" in label:
        return (label["category"], label.get("value", "POSITIVE"))
    else:
        raise ValueError(f"Couldn't parse label {label}")


def parse_labels(labels):
    parsed = []
    for label in labels:
        try:
            parsed.append(parse_label(label))
        except ValueError:
            pass
    return parsed


@click.command()
@click.argument("name")
@click.argument("label_json", type=click.File("r"))
@click.option("--user")
@click.option("--use_proxy", is_flag=True)
@unasync
async def main(name, label_json, user, use_proxy):
    if not use_proxy and ('http_proxy' in os.environ or
                          'https_proxy' in os.environ):
        print('WARNING: http_proxy/https_proxy env variables set, but '
              '--use_proxy flag not specified. Will not use proxy.')

    _, email = parseaddr(user)
    if "@" not in email:
        raise ValueError(f"Invalid email address {user}")

    raw_labels_by_path = json.load(label_json)

    # Map: turn paths into identifiers, standardize label format
    labels_by_identifier = {
        make_identifier(path): parse_labels(raw_labels)
        for path, raw_labels in raw_labels_by_path.items()
    }

    # Aggregate: group identifiers by label
    identifiers_by_label = defaultdict(list)
    for identifier, labels in labels_by_identifier.items():
        for label in labels:
            identifiers_by_label[label].append(identifier)

    # For each unique label (category, mode tuple), add annotations
    async with aiohttp.ClientSession(trust_env=use_proxy) as session:
        for (category, value), identifiers in tqdm(identifiers_by_label.items()):
            params = {
                "mode": value,
                "identifiers": identifiers,
                "user": email,
                "category": category,
                "created_by": "ingest"
            }
            async with session.post(
                os.path.join(SERVER_URL, ADD_ANNOTATIONS_ENDPOINT, name), json=params
            ) as response:
                j = await response.json()
                tqdm.write(f"Created {j['created']} ({category}, {value}) labels")


if __name__ == "__main__":
    main()
