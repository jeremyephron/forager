from collections import defaultdict, namedtuple
from dataclasses import dataclass
import distutils.util
from enum import IntEnum
import itertools
import json
import math
import os
import random
import uuid
import shutil

from typing import List, Dict, NamedTuple, Optional

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404, get_list_or_404
from django.conf import settings
from google.cloud import storage
from rest_framework.decorators import api_view
import requests
from expiringdict import ExpiringDict

from .models import Dataset, DatasetItem, Annotation, DNNModel


class LabelValue(IntEnum):
    TOMBSTONE = -1
    POSITIVE = 1
    NEGATIVE = 2
    HARD_NEGATIVE = 3
    UNSURE = 4
    CUSTOM = 5


def nest_anns(anns, nest_category=True, nest_lf=True):
    if nest_category and nest_lf:
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            c = ann.label_category
            u = ann.label_function
            data[k][c][u].append(ann)
    elif nest_category:
        data = defaultdict(lambda: defaultdict(list))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            c = ann.label_category
            data[k][c].append(ann)
    elif nest_lf:
        data = defaultdict(lambda: defaultdict(list))
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            u = ann.label_function
            data[k][u].append(ann)
    else:
        data = defaultdict(list)
        for ann in anns:
            # Only use most recent perframe ann
            k = ann.dataset_item.pk
            data[k].append(ann)
    return data


def filter_most_recent_anns(nested_anns):
    def filter_fn(anns):
        filt_anns = []
        most_recent = None
        for ann in anns:
            if ann.label_type == "klabel_frame":
                if most_recent is None or ann.created > most_recent.created:
                    most_recent = ann
            else:
                filt_anns.append(ann)
        if most_recent:
            filt_anns.append(most_recent)
        return filt_anns

    if len(nested_anns) == 0:
        return {}
    if isinstance(next(iter(nested_anns.items()))[1], list):
        data = defaultdict(list)
        for pk, anns in nested_anns.items():
            data[pk] = filter_fn(anns)
    elif isinstance(next(iter(next(iter(nested_anns.items()))[1].items()))[1], list):
        data = defaultdict(lambda: defaultdict(list))
        for pk, label_fns_data in nested_anns.items():
            for label_fn, anns in label_fns_data.items():
                data[pk][label_fn] = filter_fn(anns)
    elif isinstance(
        next(iter(next(iter(next(iter(nested_anns.items()))[1].items()))[1].items()))[
            1
        ],
        list,
    ):
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for pk, cat_fns_data in nested_anns.items():
            for cat, label_fns_data in cat_fns_data.items():
                for label_fn, anns in label_fns_data.items():
                    data[pk][cat][label_fn] = filter_fn(anns)
    return data


@api_view(["POST"])
@csrf_exempt
def start_cluster(request):
    # TODO(mihirg): Remove this setting from Django; it's now managed by Terraform
    # (or figure out how to set it from the frontend if we need that)
    params = {"n_nodes": settings.EMBEDDING_CLUSTER_NODES}
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_cluster",
        json=params,
    )
    response_data = r.json()
    return JsonResponse(
        {
            "status": "success",
            "cluster_id": response_data["cluster_id"],
        }
    )


@api_view(["GET"])
@csrf_exempt
def get_cluster_status(request, cluster_id):
    params = {"cluster_id": cluster_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/cluster_status", params=params
    )
    response_data = r.json()
    return JsonResponse(response_data)


@api_view(["POST"])
@csrf_exempt
def stop_cluster(request, cluster_id):
    params = {"cluster_id": cluster_id}
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/stop_cluster",
        json=params,
    )
    return JsonResponse(
        {
            "status": "success",
        }
    )


@api_view(["POST"])
@csrf_exempt
def create_model(request, dataset_name, dataset=None):
    payload = json.loads(request.body)
    model_name = payload["model_name"]
    cluster_id = payload["cluster_id"]
    bucket_name = payload["bucket"]
    index_id = payload["index_id"]
    pos_tags = parse_tag_set_from_query_v2(payload["pos_tags"])
    neg_tags = parse_tag_set_from_query_v2(payload["neg_tags"])
    val_pos_tags = parse_tag_set_from_query_v2(payload["val_pos_tags"])
    val_neg_tags = parse_tag_set_from_query_v2(payload["val_neg_tags"])
    augment_negs = bool(payload["augment_negs"])
    model_kwargs = payload["model_kwargs"]
    resume_model_id = payload.get("resume", None)

    dataset = get_object_or_404(Dataset, name=dataset_name)
    eligible_images = DatasetItem.objects.filter(
        dataset=dataset, google=False, is_val=False
    )
    annotations = Annotation.objects.filter(
        dataset_item__in=eligible_images,
        label_category__in=tag_sets_to_category_list_v2(
            pos_tags, neg_tags, val_pos_tags, val_neg_tags
        ),
        label_type="klabel_frame",
    )
    tags_by_pk = get_tags_from_annotations_v2(annotations)

    pos_dataset_item_pks = []
    neg_dataset_item_pks = []
    val_pos_dataset_item_pks = []
    val_neg_dataset_item_pks = []
    for pk, tags in tags_by_pk.items():
        if any(t in pos_tags for t in tags):
            pos_dataset_item_pks.append(pk)
        elif any(t in neg_tags for t in tags):
            neg_dataset_item_pks.append(pk)
        elif any(t in val_pos_tags for t in tags):
            val_pos_dataset_item_pks.append(pk)
        elif any(t in val_neg_tags for t in tags):
            val_neg_dataset_item_pks.append(pk)

    # Augment with randomly sampled negatives if requested
    num_extra_negs = settings.BGSPLIT_NUM_NEGS_MULTIPLIER * len(
        pos_dataset_item_pks
    ) - len(neg_dataset_item_pks)
    if augment_negs and num_extra_negs > 0:
        # Uses "include" and "exclude" category sets from request
        all_eligible_pks = filtered_images_v2(
            request, dataset, exclude_pks=pos_dataset_item_pks + neg_dataset_item_pks
        )
        sampled_pks = random.sample(
            all_eligible_pks, min(len(all_eligible_pks), num_extra_negs)
        )
        neg_dataset_item_pks.extend(sampled_pks)

    pos_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=pos_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    neg_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=neg_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    val_pos_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=val_pos_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    val_neg_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=val_neg_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )

    if resume_model_id:
        resume_model = get_object_or_404(DNNModel, model_id=resume_model_id)
        resume_model_path = resume_model.checkpoint_path
    else:
        resume_model = None
        resume_model_path = None

    params = {
        "pos_identifiers": pos_dataset_item_internal_identifiers,
        "neg_identifiers": neg_dataset_item_internal_identifiers,
        "val_pos_identifiers": val_pos_dataset_item_internal_identifiers,
        "val_neg_identifiers": val_neg_dataset_item_internal_identifiers,
        "augment_negs": augment_negs,
        "model_kwargs": model_kwargs,
        "model_name": model_name,
        "bucket": bucket_name,
        "cluster_id": cluster_id,
        "index_id": index_id,
        "resume_from": resume_model_path,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_bgsplit_job",
        json=params,
    )
    response_data = r.json()

    if r.status_code != 200:
        return JsonResponse(
            {"status": "failure", "reason": response_data.get("reason", "")},
            status=r.status_code,
        )

    m = DNNModel(
        dataset=dataset,
        name=model_name,
        model_id=response_data["model_id"],
        category_spec={
            "augment_negs": augment_negs,
            "pos_tags": payload["pos_tags"],
            "neg_tags": payload["neg_tags"],
            "augment_negs_include": payload.get("include", []) if augment_negs else [],
            "augment_negs_exclude": payload.get("exclude", []) if augment_negs else [],
        },
    )
    model_epoch = -1 + model_kwargs.get("epochs_to_run", 1)
    if resume_model_id:
        m.resume_model_id = resume_model_id
        if model_kwargs.get("resume_training", False):
            model_epoch += resume_model.epoch + 1
    m.epoch = model_epoch
    m.save()

    return JsonResponse(
        {
            "status": "success",
            "model_id": response_data["model_id"],
        }
    )


@api_view(["GET"])
@csrf_exempt
def get_model_status(request, model_id):
    params = {"model_id": model_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/bgsplit_job_status", params=params
    )
    response_data = r.json()
    if response_data["has_model"]:
        # Index has been successfully created & uploaded -> persist
        m = get_object_or_404(DNNModel, model_id=model_id)
        m.checkpoint_path = response_data["checkpoint_path"]
        m.save()

    return JsonResponse(response_data)


@api_view(["POST"])
@csrf_exempt
def update_model_v2(request):
    payload = json.loads(request.body)
    # user = payload["user"]
    old_model_name = payload["old_model_name"]
    new_model_name = payload["new_model_name"]

    models = get_list_or_404(DNNModel, name=old_model_name)
    for m in models:
        m.name = new_model_name
        m.save()

    return JsonResponse({"success": True})


@api_view(["POST"])
@csrf_exempt
def delete_model_v2(request):
    payload = json.loads(request.body)
    model_name = payload["model_name"]
    # cluster_id = payload['cluster_id']
    models = get_list_or_404(DNNModel, name=model_name)
    for m in models:
        # TODO(fpoms): delete model data stored on NFS?
        # shutil.rmtree(os.path.join(m.checkpoint_path, '..'))
        shutil.rmtree(m.output_directory, ignore_errors=True)
        m.delete()

    return JsonResponse({"success": True})


@api_view(["POST"])
@csrf_exempt
def run_model_inference(request, dataset_name, dataset=None):
    payload = json.loads(request.body)
    model_id = payload["model_id"]
    cluster_id = payload["cluster_id"]
    bucket_name = payload["bucket"]
    index_id = payload["index_id"]

    dataset = get_object_or_404(Dataset, name=dataset_name)
    model_checkpoint_path = get_object_or_404(
        DNNModel, model_id=model_id
    ).checkpoint_path
    if model_checkpoint_path is None or len(model_checkpoint_path) == 0:
        return JsonResponse(
            {
                "status": "failure",
                "reason": f"Model {model_id} does not have a model checkpoint.",
            },
            status=400,
        )

    params = {
        "bucket": bucket_name,
        "model_id": model_id,
        "checkpoint_path": model_checkpoint_path,
        "cluster_id": cluster_id,
        "index_id": index_id,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/start_bgsplit_inference_job",
        json=params,
    )
    response_data = r.json()

    return JsonResponse(
        {
            "status": "success",
            "job_id": response_data["job_id"],
        }
    )


@api_view(["GET"])
@csrf_exempt
def get_model_inference_status(request, job_id):
    params = {"job_id": job_id}
    r = requests.get(
        settings.EMBEDDING_SERVER_ADDRESS + "/bgsplit_inference_job_status",
        params=params,
    )
    response_data = r.json()
    if response_data["has_output"]:
        model_id = response_data["model_id"]
        # Index has been successfully created & uploaded -> persist
        m = get_object_or_404(DNNModel, model_id=model_id)
        m.output_directory = response_data["output_dir"]
        m.save()

    return JsonResponse(response_data)


@api_view(["POST"])
@csrf_exempt
def stop_model_inference(request, job_id):
    params = {"job_id": job_id}
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/stop_bgsplit_inference_job", json=params
    )
    response_data = r.json()
    return JsonResponse(response_data, status=r.status_code)


#
# V2 ENDPOINTS
# TODO(mihirg): Make these faster
#


Tag = namedtuple("Tag", "category value")  # type: NamedTuple[str, str]
PkType = int


@dataclass
class ResultSet:
    type: str
    ranking: List[PkType]
    distances: List[float]
    model: Optional[str]


# TODO(fpoms): this needs to be wrapped in a lock so that
# updates are atomic across concurrent requests
current_result_sets = ExpiringDict(
    max_age_seconds=30 * 60,
    max_len=50,
)  # type: Dict[str, ResultSet]


def parse_tag_set_from_query_v2(s):
    if isinstance(s, list):
        parts = s
    elif isinstance(s, str) and s:
        parts = s.split(",")
    else:
        parts = []
    ts = set()
    for part in parts:
        if not part:
            continue
        category, value_str = part.split(":")
        ts.add(Tag(category, value_str))
    return ts


def tag_sets_to_category_list_v2(*tagsets):
    categories = set()
    for ts in tagsets:
        for tag in ts:
            categories.add(tag.category)
    return list(categories)


def serialize_tag_set_for_client_v2(ts):
    return [{"category": t.category, "value": t.value} for t in sorted(list(ts))]


def get_tags_from_annotations_v2(annotations):
    # [image][category][#]
    anns = filter_most_recent_anns(nest_anns(annotations, nest_lf=False))

    tags_by_pk = defaultdict(list)
    for di_pk, anns_by_cat in anns.items():
        for cat, ann_list in anns_by_cat.items():
            if not cat:
                continue
            assert len(ann_list) == 1  # should only be one latest per-frame annotation
            ann = ann_list[0]

            label_data = json.loads(ann.label_data)
            value = LabelValue(label_data["value"])
            if value == LabelValue.TOMBSTONE:
                continue
            value_str = (
                label_data["custom_value"] if value == LabelValue.CUSTOM else value.name
            )
            tags_by_pk[di_pk].append(Tag(cat, value_str))
    return tags_by_pk


def filtered_images_v2(request, dataset, exclude_pks=None) -> List[PkType]:
    if request.method == "POST":
        payload = json.loads(request.body)
        include_tags = parse_tag_set_from_query_v2(payload.get("include"))
        exclude_tags = parse_tag_set_from_query_v2(payload.get("exclude"))
        pks = [i for i in payload.get("subset", []) if i]
        split = payload.get("split", "train")
    else:
        include_tags = parse_tag_set_from_query_v2(request.GET.get("include"))
        exclude_tags = parse_tag_set_from_query_v2(request.GET.get("exclude"))
        pks = [i for i in request.GET.get("subset", "").split(",") if i]
        split = request.GET.get("split", "train")

    dataset_items = None
    is_val = split == "val"

    if pks and exclude_pks:
        exclude_pks = set(exclude_pks)
        pks = [pk for pk in pks if pk not in exclude_pks]
    elif not pks:
        dataset_items = DatasetItem.objects.filter(
            dataset=dataset, google=False, is_val=is_val
        )
        if exclude_pks:
            dataset_items = dataset_items.exclude(pk__in=exclude_pks)
        pks = list(dataset_items.values_list("pk", flat=True))

    if not include_tags and not exclude_tags:
        return pks

    if dataset_items is None:
        dataset_items = DatasetItem.objects.filter(pk__in=pks)

    # TODO(mihirg, fpoms): Speed up by filtering positive labels without having to json
    # decode all annotations
    annotations = (
        Annotation.objects.filter(
            dataset_item__in=dataset_items,
            label_category__in=tag_sets_to_category_list_v2(include_tags, exclude_tags),
            label_type="klabel_frame",
        )
        # ).order_by(
        #     "dataset_item", "label_category", "-created"
        # ).distinct("dataset_item", "label_category")
    )
    tags_by_pk = get_tags_from_annotations_v2(annotations)

    # TODO(mihirg, fpoms): Move all logic here into SQL query
    filtered = []
    for pk in pks:
        include = not include_tags  # include everything if no explicit filter
        exclude = False
        for tag in tags_by_pk[pk]:
            if tag in exclude_tags:
                exclude = True
                break
            elif tag in include_tags:
                include = True

        if include and not exclude:
            filtered.append(pk)

    return filtered


def process_image_query_results_v2(request, dataset, query_response):
    filtered_pks = filtered_images_v2(request, dataset)
    # TODO(mihirg): Eliminate this database call by directly returning pks from backend
    dataset_items = DatasetItem.objects.filter(pk__in=filtered_pks)
    dataset_items_by_path = {di.path: di for di in dataset_items}

    distances = []
    ordered_pks = []
    for r in query_response["results"]:
        if r["label"] in dataset_items_by_path:
            ordered_pks.append(dataset_items_by_path[r["label"]].pk)
            distances.append(r["dist"])
    return dict(
        pks=ordered_pks,
        distances=distances,
    )


def create_result_set_v2(results, type, model=None):
    pks = results["pks"]
    distances = results["distances"]
    result_set_id = str(uuid.uuid4())
    current_result_sets[result_set_id] = ResultSet(
        type=type, ranking=pks, distances=distances, model=model
    )
    return {
        "id": result_set_id,
        "num_results": len(pks),
        "type": type,
    }


@api_view(["GET"])
@csrf_exempt
def get_results_v2(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    index_id = request.GET["index_id"]
    result_set_id = request.GET["result_set_id"]
    offset_to_return = int(request.GET.get("offset", 0))
    num_to_return = int(request.GET.get("num", 1000))

    result_set = current_result_sets[result_set_id]
    pks = result_set.ranking[offset_to_return : offset_to_return + num_to_return]
    distances = result_set.distances[
        offset_to_return : offset_to_return + num_to_return
    ]

    dataset_items_by_pk = DatasetItem.objects.in_bulk(pks)
    dataset_items = [dataset_items_by_pk[pk] for pk in pks]  # preserve order

    bucket_name = dataset.directory[len("gs://") :].split("/")[0]
    path_template = "https://storage.googleapis.com/{:s}/".format(bucket_name) + "{:s}"

    internal_identifiers = [di.identifier for di in dataset_items]
    params = {
        "index_id": index_id,
        "identifiers": internal_identifiers,
    }
    if False and result_set.model:
        params["model"] = result_set.model

    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/perform_clustering",
        json=params,
    )
    clustering_data = r.json()

    dataset_item_paths = [
        (di.path if di.path.find("http") != -1 else path_template.format(di.path))
        for di in dataset_items
    ]
    dataset_item_identifiers = [di.pk for di in dataset_items]

    return JsonResponse(
        {
            "paths": dataset_item_paths,
            "identifiers": dataset_item_identifiers,
            "distances": distances,
            "clustering": clustering_data["clustering"],
        }
    )


@api_view(["POST"])
@csrf_exempt
def keep_alive_v2(request):
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/keep_alive",
    )
    return JsonResponse({"status": "success"})


@api_view(["POST"])
@csrf_exempt
def generate_embedding_v2(request):
    payload = json.loads(request.body)
    image_id = payload.get("image_id")
    if image_id:
        payload["identifier"] = DatasetItem.objects.get(pk=image_id).identifier

    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/generate_embedding",
        json=payload,
    )
    return JsonResponse(r.json())


@api_view(["POST"])
@csrf_exempt
def generate_text_embedding_v2(request):
    payload = json.loads(request.body)
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/generate_text_embedding",
        json=payload,
    )
    return JsonResponse(r.json())


@api_view(["POST"])
@csrf_exempt
def query_knn_v2(request, dataset_name):
    payload = json.loads(request.body)
    index_id = payload["index_id"]
    embeddings = payload["embeddings"]
    use_full_image = bool(payload.get("use_full_image", True))
    use_dot_product = bool(payload.get("use_dot_product", False))
    model = payload.get("model", "imagenet")

    dataset = get_object_or_404(Dataset, name=dataset_name)

    params = {
        "index_id": index_id,
        "embeddings": embeddings,
        "use_full_image": use_full_image,
        "use_dot_product": use_dot_product,
        "model": model,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_knn_v2",
        json=params,
    )
    response_data = r.json()

    results = process_image_query_results_v2(
        request,
        dataset,
        response_data,
    )
    return JsonResponse(create_result_set_v2(results, "knn", model=model))


@api_view(["GET"])
@csrf_exempt
def train_svm_v2(request, dataset_name):
    index_id = request.GET["index_id"]
    model = request.GET.get("model", "imagenet")
    pos_tags = parse_tag_set_from_query_v2(request.GET["pos_tags"])
    neg_tags = parse_tag_set_from_query_v2(request.GET.get("neg_tags"))
    augment_negs = bool(
        distutils.util.strtobool(request.GET.get("augment_negs", "false"))
    )

    dataset = get_object_or_404(Dataset, name=dataset_name)
    eligible_images = DatasetItem.objects.filter(
        dataset=dataset, google=False, is_val=False
    )
    annotations = Annotation.objects.filter(
        dataset_item__in=eligible_images,
        label_category__in=tag_sets_to_category_list_v2(pos_tags, neg_tags),
        label_type="klabel_frame",
    )
    tags_by_pk = get_tags_from_annotations_v2(annotations)

    pos_dataset_item_pks = []
    neg_dataset_item_pks = []
    for pk, tags in tags_by_pk.items():
        if any(t in pos_tags for t in tags):
            pos_dataset_item_pks.append(pk)
        elif any(t in neg_tags for t in tags):
            neg_dataset_item_pks.append(pk)

    # Augment with randomly sampled negatives if requested
    num_extra_negs = settings.SVM_NUM_NEGS_MULTIPLIER * len(pos_dataset_item_pks) - len(
        neg_dataset_item_pks
    )
    if augment_negs and num_extra_negs > 0:
        # Uses "include" and "exclude" category sets from GET request
        all_eligible_pks = filtered_images_v2(
            request, dataset, exclude_pks=pos_dataset_item_pks + neg_dataset_item_pks
        )
        sampled_pks = random.sample(
            all_eligible_pks, min(len(all_eligible_pks), num_extra_negs)
        )
        neg_dataset_item_pks.extend(sampled_pks)

    pos_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=pos_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )
    neg_dataset_item_internal_identifiers = list(
        DatasetItem.objects.filter(pk__in=neg_dataset_item_pks).values_list(
            "identifier", flat=True
        )
    )

    params = {
        "index_id": index_id,
        "pos_identifiers": pos_dataset_item_internal_identifiers,
        "neg_identifiers": neg_dataset_item_internal_identifiers,
        "model": model,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/train_svm_v2",
        json=params,
    )
    return JsonResponse(r.json())  # {"svm_vector": base64-encoded string}


@api_view(["POST"])
@csrf_exempt
def query_svm_v2(request, dataset_name):
    payload = json.loads(request.body)
    index_id = payload["index_id"]
    svm_vector = payload["svm_vector"]
    score_min = float(payload.get("score_min", 0.0))
    score_max = float(payload.get("score_max", 1.0))
    model = payload.get("model", "imagenet")

    dataset = get_object_or_404(Dataset, name=dataset_name)

    params = {
        "index_id": index_id,
        "svm_vector": svm_vector,
        "score_min": score_min,
        "score_max": score_max,
        "model": model,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_svm_v2",
        json=params,
    )
    response_data = r.json()

    # TODO(mihirg, jeremye): Consider some smarter pagination/filtering scheme to avoid
    # running a separate query over the index every single time the user adjusts score
    # thresholds
    results = process_image_query_results_v2(
        request,
        dataset,
        response_data,
    )
    return JsonResponse(create_result_set_v2(results, "svm"))


@api_view(["POST"])
@csrf_exempt
def query_ranking_v2(request, dataset_name):
    payload = json.loads(request.body)
    index_id = payload["index_id"]
    score_min = float(payload.get("score_min", 0.0))
    score_max = float(payload.get("score_max", 1.0))
    model = payload["model"]

    dataset = get_object_or_404(Dataset, name=dataset_name)

    params = {
        "index_id": index_id,
        "score_min": score_min,
        "score_max": score_max,
        "model": model,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_ranking_v2",
        json=params,
    )
    response_data = r.json()

    # TODO(mihirg, jeremye): Consider some smarter pagination/filtering scheme to avoid
    # running a separate query over the index every single time the user adjusts score
    # thresholds
    results = process_image_query_results_v2(
        request,
        dataset,
        response_data,
    )
    return JsonResponse(create_result_set_v2(results, "ranking", model=model))


@api_view(["POST"])
@csrf_exempt
def query_images_v2(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    payload = json.loads(request.body)
    order = payload.get("order", "id")

    result_pks = filtered_images_v2(request, dataset)
    if order == "random":
        random.shuffle(result_pks)
    elif order == "id":
        result_pks.sort()
    results = {"pks": result_pks, "distances": [-1 for _ in result_pks]}
    return JsonResponse(create_result_set_v2(results, "query"))


#
# ACTIVE VALIDATION
#


VAL_NEGATIVE_TYPE = "model_val_negative"


def get_val_examples_v2(dataset, model_id):
    # Get positive and negative categories
    model = get_object_or_404(DNNModel, model_id=model_id)

    pos_tags = parse_tag_set_from_query_v2(model.category_spec["pos_tags"])
    neg_tags = parse_tag_set_from_query_v2(model.category_spec["neg_tags"])
    augment_negs = model.category_spec.get("augment_negs", False)
    augment_negs_include = (
        parse_tag_set_from_query_v2(model.category_spec.get("augment_negs_include", []))
        if augment_negs
        else set()
    )

    # Limit to validation set
    eligible_dataset_items = DatasetItem.objects.filter(
        dataset=dataset,
        google=False,
        is_val=True,
    )

    # Get positives and negatives matching these categories
    annotations = Annotation.objects.filter(
        dataset_item__in=eligible_dataset_items,
        label_category__in=tag_sets_to_category_list_v2(
            pos_tags, neg_tags, augment_negs_include
        ),
        label_type="klabel_frame",
    )
    tags_by_pk = get_tags_from_annotations_v2(annotations)

    pos_dataset_item_pks = []
    neg_dataset_item_pks = []
    for pk, tags in tags_by_pk.items():
        if any(t in pos_tags for t in tags):
            pos_dataset_item_pks.append(pk)
        elif any(t in neg_tags or t in augment_negs_include for t in tags):
            neg_dataset_item_pks.append(pk)

    # Get extra negatives
    if augment_negs:
        annotations = Annotation.objects.filter(
            dataset_item__in=eligible_dataset_items,
            label_category=model_id,
            label_type=VAL_NEGATIVE_TYPE,
        )
        neg_dataset_item_pks.extend(ann.dataset_item.pk for ann in annotations)

    return pos_dataset_item_pks, neg_dataset_item_pks


@api_view(["POST"])
def query_metrics_v2(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    payload = json.loads(request.body)
    model_id = payload["model"]
    index_id = payload["index_id"]
    internal_identifiers_to_weights = payload["weights"]  # type: Dict[str, int]

    pos_dataset_item_pks, neg_dataset_item_pks = get_val_examples_v2(dataset, model_id)

    # Construct identifiers, labels, and weights
    dataset_items_by_pk = DatasetItem.objects.in_bulk(
        pos_dataset_item_pks + neg_dataset_item_pks
    )
    identifiers = []
    labels = []
    weights = []
    for pk, label in itertools.chain(
        ((pk, True) for pk in pos_dataset_item_pks),
        ((pk, False) for pk in neg_dataset_item_pks),
    ):
        di = dataset_items_by_pk[pk]
        identifier = di.identifier
        weight = internal_identifiers_to_weights.get(identifier)
        if weight is None:
            continue

        identifiers.append(identifier)
        labels.append(label)
        weights.append(weight)

    # TODO(mihirg): Parse false positives and false negatives
    params = {
        "index_id": index_id,
        "model": model_id,
        "identifiers": identifiers,
        "labels": labels,
        "weights": weights,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_metrics",
        json=params,
    )
    response_data = r.json()
    return JsonResponse(response_data)


@api_view(["POST"])
def query_active_validation_v2(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    payload = json.loads(request.body)
    model_id = payload["model"]
    index_id = payload["index_id"]
    current_f1 = payload.get("current_f1")
    if current_f1 is None:
        current_f1 = 0.5

    pos_dataset_item_pks, neg_dataset_item_pks = get_val_examples_v2(dataset, model_id)

    # Construct paths, identifiers, and labels
    dataset_items_by_pk = DatasetItem.objects.in_bulk(
        pos_dataset_item_pks + neg_dataset_item_pks
    )
    identifiers = []
    labels = []
    for pk, label in itertools.chain(
        ((pk, True) for pk in pos_dataset_item_pks),
        ((pk, False) for pk in neg_dataset_item_pks),
    ):
        di = dataset_items_by_pk[pk]
        identifiers.append(di.identifier)
        labels.append(label)

    params = {
        "index_id": index_id,
        "model": model_id,
        "identifiers": identifiers,
        "labels": labels,
        "current_f1": current_f1,
    }
    r = requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/query_active_validation",
        json=params,
    )
    response_data = r.json()

    if response_data["identifiers"]:
        pks_and_paths = list(
            DatasetItem.objects.filter(
                dataset=dataset,
                identifier__in=response_data["identifiers"],
                is_val=True,
                google=False,
            ).values_list("pk", "path")
        )
        random.shuffle(pks_and_paths)
        pks, paths = zip(*pks_and_paths)
    else:
        pks, paths = [], []

    bucket_name = dataset.val_directory[len("gs://") :].split("/")[0]
    path_template = "https://storage.googleapis.com/{:s}/".format(bucket_name) + "{:s}"
    paths = [path_template.format(p) for p in paths]

    return JsonResponse(
        {
            "paths": paths,
            "identifiers": pks,
            "weights": response_data["weights"],
        }
    )


@api_view(["POST"])
def add_val_annotations_v2(request):
    payload = json.loads(request.body)
    annotations = payload["annotations"]
    user = payload["user"]
    model = payload["model"]

    anns = []
    for ann_payload in annotations:
        image_pk = ann_payload["identifier"]
        is_other_negative = ann_payload.get("is_other_negative", False)
        value_str = "NEGATIVE" if is_other_negative else ann_payload["value"]
        category = model if is_other_negative else ann_payload["category"]
        try:
            value, custom_value = int(LabelValue[value_str]), None
        except KeyError:
            value, custom_value = LabelValue.CUSTOM, value_str

        annotation_data = {
            "type": 0,  # full-frame
            "value": value,
            "mode": "val",
            "version": ANN_VERSION,
        }
        if custom_value:
            annotation_data["custom_value"] = custom_value
        annotation = json.dumps(annotation_data)

        di = DatasetItem.objects.get(pk=image_pk)
        assert not di.google and di.is_val
        ann = Annotation(
            dataset_item=di,
            label_function=user,
            label_category=category,
            label_type=VAL_NEGATIVE_TYPE if is_other_negative else "klabel_frame",
            label_data=annotation,
        )
        anns.append(ann)

    Annotation.objects.bulk_create(anns)
    return JsonResponse({"created": len(anns)})


# DATASET INFO


@api_view(["GET"])
@csrf_exempt
def get_datasets_v2(request):
    datasets = Dataset.objects.filter(hidden=False)
    dataset_names = list(datasets.values_list("name", flat=True))
    return JsonResponse({"dataset_names": dataset_names})


@api_view(["GET"])
@csrf_exempt
def get_dataset_info_v2(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    annotations = (
        Annotation.objects.filter(
            dataset_item__in=dataset.datasetitem_set.filter(),
            label_type__exact="klabel_frame",  # deliberately exclude VAL_NEGATIVE_TYPE
        )
        .order_by("dataset_item", "label_category", "-created")
        .distinct("dataset_item", "label_category")
    )

    # Unique categories that have at least one non-tombstone label
    categories_and_custom_values = {}
    for ann in annotations:
        label_data = json.loads(ann.label_data)
        if label_data["value"] == LabelValue.TOMBSTONE:
            continue
        custom_value_set = categories_and_custom_values.setdefault(
            ann.label_category, set()
        )
        if "custom_value" in label_data:
            custom_value_set.add(label_data["custom_value"])
    categories_and_custom_values = {
        k: sorted(v) for k, v in categories_and_custom_values.items()
    }

    return JsonResponse(
        {
            "categories": categories_and_custom_values,
            "index_id": dataset.index_id,
            "num_train": dataset.datasetitem_set.filter(
                google=False, is_val=False
            ).count(),
            "num_val": dataset.datasetitem_set.filter(
                google=False, is_val=True
            ).count(),
        }
    )


@api_view(["GET"])
@csrf_exempt
def get_models_v2(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)

    model_objs = DNNModel.objects.filter(
        dataset=dataset,
        checkpoint_path__isnull=False,
    ).order_by("-last_updated")

    model_names = set()
    latest = {}
    with_output = {}
    for model in model_objs:
        model_names.add(model.name)
        if model.name not in latest:
            latest[model.name] = model
        if model.output_directory and model.name not in with_output:
            with_output[model.name] = model

    models = [
        {
            "name": model_name,
            "latest": model_info(latest[model_name]),
            "with_output": model_info(with_output.get(model_name)),
        }
        for model_name in model_names
    ]
    return JsonResponse({"models": models})


def model_info(model):
    if model is None:
        return None

    pos_tags = parse_tag_set_from_query_v2(model.category_spec.get("pos_tags", []))
    neg_tags = parse_tag_set_from_query_v2(model.category_spec.get("neg_tags", []))
    augment_negs_include = parse_tag_set_from_query_v2(
        model.category_spec.get("augment_negs_include", [])
    )
    return {
        "model_id": model.model_id,
        "timestamp": model.last_updated,
        "has_checkpoint": model.checkpoint_path is not None,
        "has_output": model.output_directory is not None,
        "pos_tags": serialize_tag_set_for_client_v2(pos_tags),
        "neg_tags": serialize_tag_set_for_client_v2(neg_tags | augment_negs_include),
        "augment_negs": model.category_spec.get("augment_negs", False),
        "epoch": model.epoch,
    }


@api_view(["POST"])
@csrf_exempt
def create_dataset_v2(request):
    payload = json.loads(request.body)
    name = payload["dataset"]
    train_directory = payload["train_path"]
    val_directory = payload["val_path"]
    index_id = payload["index_id"]

    assert all(d.startswith("gs://") for d in (train_directory, val_directory))

    # Download index on index server
    params = {"index_id": index_id}
    requests.post(
        settings.EMBEDDING_SERVER_ADDRESS + "/download_index",
        json=params,
    )

    client = storage.Client()
    all_blobs = []

    for d, is_val in ((train_directory, False), (val_directory, True)):
        split_dir = d[len("gs://") :].split("/")
        bucket_name = split_dir[0]
        bucket_path = "/".join(split_dir[1:])

        all_blobs.extend(
            (blob, is_val)
            for blob in client.list_blobs(bucket_name, prefix=bucket_path)
        )

    dataset = Dataset(
        name=name,
        directory=train_directory,
        val_directory=val_directory,
        index_id=index_id,
    )
    dataset.save()

    # Create all the DatasetItems for this dataset
    items = [
        DatasetItem(
            dataset=dataset,
            identifier=os.path.splitext(os.path.basename(blob.name))[0],
            path=blob.name,
            is_val=is_val,
        )
        for blob, is_val in all_blobs
        if (
            blob.name.endswith(".jpg")
            or blob.name.endswith(".jpeg")
            or blob.name.endswith(".png")
        )
    ]
    DatasetItem.objects.bulk_create(items, batch_size=10000)

    return JsonResponse({"status": "success"})


@api_view(["GET"])
@csrf_exempt
def get_annotations_v2(request):
    image_identifiers = [i for i in request.GET["identifiers"].split(",") if i]
    if not image_identifiers:
        return JsonResponse({})

    annotations = Annotation.objects.filter(
        dataset_item__in=DatasetItem.objects.filter(pk__in=image_identifiers),
        label_type__in=("klabel_frame", VAL_NEGATIVE_TYPE),
    )
    # ).order_by(
    #     "dataset_item", "label_category", "-created"
    # ).distinct("dataset_item", "label_category")
    tags_by_pk = get_tags_from_annotations_v2(annotations)
    annotations_by_pk = {
        pk: serialize_tag_set_for_client_v2(tags) for pk, tags in tags_by_pk.items()
    }
    return JsonResponse(annotations_by_pk)


# TODO(mihirg): Consider filtering by, e.g., major version in methods that query
# Annotations
ANN_VERSION = "2.0.2"


@api_view(["POST"])
@csrf_exempt
def add_annotations_v2(request):
    payload = json.loads(request.body)
    image_identifiers = payload["identifiers"]
    num_created = bulk_add_annotations_v2(payload, image_identifiers)
    return JsonResponse({"created": num_created})


@api_view(["POST"])
@csrf_exempt
def add_annotations_by_internal_identifiers_v2(request, dataset_name):
    # TODO(mihirg): Make variable naming more clear
    # "identifier" = di.pk (above), "internal identifier" = di.identifier (here)
    dataset = get_object_or_404(Dataset, name=dataset_name)

    payload = json.loads(request.body)
    internal_identifiers = payload["identifiers"]
    dataset_items = DatasetItem.objects.filter(
        dataset=dataset, identifier__in=internal_identifiers
    )
    pks = list(dataset_items.values_list("pk", flat=True))

    num_created = bulk_add_annotations_v2(payload, pks)
    return JsonResponse({"created": num_created})


@api_view(["POST"])
@csrf_exempt
def add_annotations_to_result_set_v2(request):
    payload = json.loads(request.body)
    result_set_id = payload["result_set_id"]
    lower_bound = float(payload["from"])
    upper_bound = float(payload["to"])

    result_set = current_result_sets[result_set_id]
    result_ranking = result_set.ranking
    # e.g., lower_bound=0.0, upper_bound=0.5 -> second half of the result set
    start_index = math.ceil(len(result_ranking) * (1.0 - upper_bound))
    end_index = math.floor(len(result_ranking) * (1.0 - lower_bound))
    image_identifiers = result_ranking[start_index:end_index]

    num_created = bulk_add_annotations_v2(payload, image_identifiers)
    return JsonResponse({"created": num_created})


def bulk_add_annotations_v2(payload, image_identifiers):
    if not image_identifiers:
        return 0

    user = payload["user"]
    category = payload["category"]
    value_str = payload["value"]
    mode = payload.get("mode", "tag" if len(image_identifiers) == 1 else "tag-bulk")
    try:
        value, custom_value = int(LabelValue[value_str]), None
    except KeyError:
        value, custom_value = LabelValue.CUSTOM, value_str

    annotation_data = {
        "type": 0,  # full-frame
        "value": value,
        "mode": mode,
        "version": ANN_VERSION,
    }
    if custom_value:
        annotation_data["custom_value"] = custom_value
    annotation = json.dumps(annotation_data)

    dataset_items = DatasetItem.objects.filter(pk__in=image_identifiers)
    Annotation.objects.bulk_create(
        (
            Annotation(
                dataset_item=di,
                label_function=user,
                label_category=category,
                label_type="klabel_frame",
                label_data=annotation,
            )
            for di in dataset_items
        )
    )

    return len(image_identifiers)


@api_view(["POST"])
@csrf_exempt
def delete_category_v2(request):
    payload = json.loads(request.body)
    # user = payload["user"]
    category = payload["category"]

    n, _ = Annotation.objects.filter(
        # label_function__exact=user,
        label_category__exact=category,
    ).delete()

    return JsonResponse({"deleted": n})


@api_view(["POST"])
@csrf_exempt
def update_category_v2(request):
    payload = json.loads(request.body)
    # user = payload["user"]
    old_category = payload["oldCategory"]
    new_category = payload["newCategory"]

    n = Annotation.objects.filter(
        # label_function__exact=user,
        label_category__exact=old_category,
    ).update(label_category=new_category)

    return JsonResponse({"updated": n})


@api_view(["POST"])
@csrf_exempt
def get_category_counts_v2(request, dataset_name):
    dataset = get_object_or_404(Dataset, name=dataset_name)
    payload = json.loads(request.body)
    categories = payload["categories"]

    anns = (
        Annotation.objects.filter(
            dataset_item__in=dataset.datasetitem_set.filter(),
            label_category__in=categories,
            label_type__exact="klabel_frame",
        )
        .order_by("dataset_item", "label_category", "-created")
        .distinct("dataset_item", "label_category")
    )

    n_labeled = {c: {value.name: 0 for value in LabelValue} for c in categories}
    for ann in anns:
        label_data = json.loads(ann.label_data)
        value = LabelValue(label_data["value"])
        n_labeled[ann.label_category][value.name] += 1

    return JsonResponse({"numLabeled": n_labeled})
