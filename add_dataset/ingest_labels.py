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
ADD_MULTI_ANNOTATIONS_ENDPOINT = "api/add_annotations_multi_v2"
BOX_BATCH_SIZE = 10000


def parse_label(label):
    if isinstance(label, str):
        return (label, "POSITIVE")
    elif isinstance(label, dict) and "is_box" in label:
        return label
    elif isinstance(label, dict) and "category" in label:
        return (label["category"], label.get("mode", "POSITIVE"))
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

    # Aggregate + split: group identifiers by label and split bbox labels out
    identifiers_by_label = defaultdict(list)
    box_annotations = []
    for identifier, labels in labels_by_identifier.items():
        for label in labels:
            if 'is_box' in label and label['is_box']:
                label['identifier'] = identifier
                box_annotations.append(label)
            else:
                identifiers_by_label[label].append(identifier)

    # For each unique label (category, mode tuple), add annotations
    async with aiohttp.ClientSession(trust_env=use_proxy) as session:
        print('Adding frame annotations...')
        for (category, mode), identifiers in tqdm(identifiers_by_label.items()):
            break
            params = {
                "mode": mode,
                "identifiers": identifiers,
                "user": email,
                "category": category,
                "created_by": "ingest"
            }
            async with session.post(
                    os.path.join(SERVER_URL, ADD_ANNOTATIONS_ENDPOINT, name),
                    json=params
            ) as response:
                j = await response.json()
                tqdm.write(f"Created {j['created']} ({category}, {mode}) labels")
        # Add all bbox annotations at once
        print('Adding box annotations...')
        total_anns = 0
        for i in tqdm(range(0, len(box_annotations), BOX_BATCH_SIZE)):
            params = {
                "dataset": name,
                "user": email,
                "annotations": box_annotations[i:i+BOX_BATCH_SIZE],
                "created_by": "ingest"
            }
            async with session.post(
                    os.path.join(SERVER_URL, ADD_MULTI_ANNOTATIONS_ENDPOINT),
                    json=params
            ) as response:
                j = await response.json()
                total_anns += j['created']
                tqdm.write(f"Created {j['created']} box labels")
        print(f"Finished creating {total_anns} total box labels")


if __name__ == "__main__":
    main()
