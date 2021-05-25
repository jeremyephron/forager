from collections import defaultdict
from email.utils import parseaddr
import os
import json

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
@unasync
async def main(name, label_json, user):
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
    async with aiohttp.ClientSession() as session:
        for (category, value), identifiers in identifiers_by_label.items():
            params = {
                "mode": "ingest",
                "identifiers": identifiers,
                "user": email,
                "category": category,
                "value": value,
            }
            async with session.post(
                os.path.join(SERVER_URL, ADD_ANNOTATIONS_ENDPOINT, name), json=params
            ) as response:
                j = await response.json()
                print(f"Created {j['created']} ({category}, {value}) labels")


if __name__ == "__main__":
    main()
