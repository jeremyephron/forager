import json
from pathlib import Path

EXISTING_IMAGE_PATHS_FILENAME = "labels_old.json"
ADDITIONAL_IMAGE_PATHS_FILENAME = "waymo-val-r50/images.json"

NEW_IDENTIFIERS_OUTPUT_FILENAME = "val_identifiers.json"
NEW_IMAGE_LIST_OUTPUT_FILENAME = "labels.json"

PATH_START_SUBSTRING = "waymo"

existing = json.load(Path(EXISTING_IMAGE_PATHS_FILENAME).open())
additional = json.load(Path(ADDITIONAL_IMAGE_PATHS_FILENAME).open())
new = existing + [p[p.find(PATH_START_SUBSTRING) :] for p in additional]
json.dump(new, Path(NEW_IMAGE_LIST_OUTPUT_FILENAME).open("w"))

make_identifier = lambda p: p[p.rfind("/") + 1 : p.rfind(".")]
new_identifiers = {
    make_identifier(p): i + len(existing) for i, p in enumerate(additional)
}
json.dump(new_identifiers, Path(NEW_IDENTIFIERS_OUTPUT_FILENAME).open("w"))
