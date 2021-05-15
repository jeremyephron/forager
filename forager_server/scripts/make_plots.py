import matplotlib.pyplot as plt
import uuid
import numpy as np

OUTPUT_FILENAME = "graph.png"

ACTIVITY_TO_COLOR = {
    "EXTERNAL BOOTSTRAPPING": "tab:blue",
    "INTERNAL KNN": "tab:orange",
    "RANDOM": "tab:green",
    "RANK": "tab:red",
    "SVM": "tab:purple",
    "VALIDATION": "tab:pink",
    "CLIP": "tab:cyan",
    "DATASET": "tab:olive",
}

ACTIVITY_FILENAME = "activity.txt"
ACCURACY_FILENAME = "accuracy.txt"
LABELS_FILENAME = "labels.txt"

accuracy = []
# with open(ACCURACY_FILENAME, "r") as f:
#     accuracy = [float(l.strip()) for l in f.readlines()]

with open(LABELS_FILENAME, "r") as f:
    labels = [(int(p), int(n)) for p, n in map(lambda x: x.split(), f.readlines())]

activity = []
is_training = []
with open(ACTIVITY_FILENAME, "r") as f:
    for l in f:
        is_training.append(l.strip().endswith("*"))
        activity.append(l.strip().rstrip("*"))

activity.append(None)

fig, axs = plt.subplots(3, sharex=True)

# for i, t in enumerate(is_training):
#     if t:
#         axs[0].barh(
#             ["Training"],
#             [1 / 60],
#             left=[i / 60],
#             height=0.5,
#             label=str(uuid.uuid4()),
#             color="tab:gray",
#         )

last_start = 0
last_activity = activity[0]
width = 0

artists_by_activity = {}

for i, a in enumerate(activity):
    if a == last_activity:
        width += 1
    else:
        artist = axs[0].barh(
            ["Activity"],
            [width / 60],
            left=[last_start / 60],
            height=0.5,
            label=str(uuid.uuid4()),
            color=ACTIVITY_TO_COLOR[last_activity],
        )
        artists_by_activity[last_activity] = artist
        width = 1
        last_start = i
        last_activity = a

axs[0].legend(
    [
        artists_by_activity[a]
        for a in ACTIVITY_TO_COLOR.keys()
        if a in artists_by_activity
    ],
    [a.title() for a in ACTIVITY_TO_COLOR.keys() if a in artists_by_activity],
    fontsize="small",
    bbox_to_anchor=(0.15, 1.0),
    ncol=3,
)

axs[1].plot(np.arange(len(accuracy)) / 60, accuracy)
axs[1].set_ylim(0, 1)
axs[1].set_ylabel("F1 score")

axs[2].plot(np.arange(len(labels)) / 60, [p for p, n in labels], label="Pos")
axs[2].plot(np.arange(len(labels)) / 60, [n for p, n in labels], label="Neg")
axs[2].set_ylabel("# labels")
axs[2].legend()

axs[2].set_xlabel("Time (min)")

plt.savefig(OUTPUT_FILENAME, dpi=300)
