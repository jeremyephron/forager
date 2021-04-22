import React, { useState, useEffect, useCallback, useReducer, useRef } from "react";
import useInterval from 'react-useinterval';
import {
  Container,
  Row,
  Col,
  Button,
  Nav,
  Navbar,
  NavItem,
  NavLink,
  NavbarBrand,
  FormGroup,
  Input,
  Modal,
  ModalBody,
  Popover,
  PopoverBody,
  Spinner,
  Collapse,
} from "reactstrap";
import { Typeahead } from "react-bootstrap-typeahead";
import { ReactSVG } from "react-svg";
import Slider, { Range } from "rc-slider";
import Emoji from "react-emoji-render";
import ReactTimeAgo from "react-time-ago";
import { v4 as uuidv4 } from "uuid";
import { faCog, faChevronUp } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import fromPairs from "lodash/fromPairs";
import size from "lodash/size";
import some from "lodash/some";
import sortBy from "lodash/sortBy";
import toPairs from "lodash/toPairs";

import "react-bootstrap-typeahead/css/Typeahead.css";
import "rc-slider/assets/index.css";
import "./scss/theme.scss";

import {
  ClusterModal,
  ImageStack,
  SignInModal,
  TagManagementModal,
  CategoryInput,
  FeatureInput,
  NewModeInput,
  KnnPopover,
  ModelRankingPopover,
  CaptionSearchPopover,
} from "./components";

var disjointSet = require("disjoint-set");
var dateFormat = require("dateformat");

// TODO(mihirg): Combine with this same constant in other places
const LABEL_VALUES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const orderingModes = [
  {id: "random", label: "Random order"},
  {id: "id", label: "Dataset order"},
  {id: "svm", label: "SVM"},
  {id: "knn", label: "KNN"},
  {id: "dnn", label: "Model ranking"},
  {id: "clip", label: "Caption search"},
];

const dnns = [
  {id: "dnn", label: "DNN w/ BG Splitting"},
];

const endpoints = fromPairs(toPairs({
  getDatasetInfo: "get_dataset_info_v2",
  getModels: "get_models_v2",
  getNextImages: "get_next_images_v2",
  trainSvm: "train_svm_v2",
  querySvm: "query_svm_v2",
  queryKnn: "query_knn_v2",
  queryRanking: "query_ranking_v2",
  trainModel: "train_model_v2",
  modelStatus: "model_v2",
  modelInference: "model_inference_v2",
  modelInferenceStatus: "model_inference_status_v2",
  stopModelInference: "stop_model_inference_v2",
  startCluster: "start_cluster",
  clusterStatus: "cluster",
  generateEmbedding: 'generate_embedding_v2',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const App = () => {
  //
  // DOCUMENT EVENT HANDLERS
  //

  const [hasDrag, setHasDrag] = useState(false);
  const dragRefCount = useRef(0);

  const onDragEnter = () => {
    dragRefCount.current = dragRefCount.current + 1;
    setHasDrag(true);
  };

  const onDragExit = () => {
    dragRefCount.current = dragRefCount.current - 1;
    if (dragRefCount.current === 0) setHasDrag(false);
  };

  const onDrop = () => {
    dragRefCount.current = 0;
    setHasDrag(false);
  };

  useEffect(() => {
    window.onbeforeunload = () => "Are you sure you want to exit Forager?";
    document.addEventListener("dragenter", onDragEnter);
    document.addEventListener("dragleave", onDragExit);
    document.addEventListener("drop", onDrop);
    return () => {
      document.removeEventListener("dragenter", onDragEnter);
      document.removeEventListener("dragleave", onDragExit);
      document.removeEventListener("drop", onDrop);
    }
  }, [onDragEnter, onDragExit, onDrop]);

  //
  // USER AUTHENTICATION
  //

  const [username, setUsername_] = useState(
    window.localStorage.getItem("foragerUsername") || ""
  );
  const [loginIsOpen, setLoginIsOpen] = useState(false);
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  const setUsername = (u) => {
    window.localStorage.setItem("foragerUsername", u);
    setUsername_(u);
  }

  const login = (e) => {
    if (loginUsername !== undefined && loginPassword === "forager") setUsername(loginUsername.trim());
    setLoginIsOpen(false);
    e.preventDefault();
  }

  //
  // CLUSTER FOCUS MODAL
  //

  const [selection, setSelection] = useState({});
  const [clusterIsOpen, setClusterIsOpen] = useState(false);

  //
  // TAG MANAGEMENT MODAL
  //
  const [tagManagementIsOpen, setTagManagementIsOpen] = useState(false);
  const toggleTagManagement = () => setTagManagementIsOpen(!tagManagementIsOpen);

  //
  // DATA CONNECTIONS
  //

  // Load dataset info on initial page load
  const [datasetName, setDatasetName] = useState("waymo_train_central");
  const [datasetInfo, setDatasetInfo] = useState({
    isNotLoaded: true,
    categories: [],
    index_id: null,
    num_images: 0,
    num_google: 0,
  });
  const [popoverOpen, setPopoverOpen] = useState(false);

  useEffect(() => {
    const getDatasetInfo = async () => {
      const url = new URL(`${endpoints.getDatasetInfo}/${datasetName}`);
      let _datasetInfo = await fetch(url, {
        method: "GET",
      }).then(r => r.json());
      setDatasetInfo(_datasetInfo);
      setIsLoading(true);
    }
    getDatasetInfo();
  }, [datasetName]);

  const setCategories = (categories) => setDatasetInfo({...datasetInfo, categories});

  // KNN queries
  const generateEmbedding = async (req, uuid) => {
    const url = new URL(endpoints.generateEmbedding);
    const body = {
      index_id: datasetInfo.index_id,
      ...req,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    knnImagesDispatch({
      type: "SET_EMBEDDING",
      embedding: res.embedding,
      uuid,
    });
  };

  const knnReducer = (state, action) => {
    switch (action.type) {
      case "ADD_IMAGE_FROM_DATASET": {
        let newState = {...state};
        let newImageState = {type: "dataset", id: action.image.id, src: action.image.thumb};
        newState[action.uuid] = newImageState;
        return newState;
      }
      case "ADD_IMAGE_FILE": {
        let newState = {...state};
        let newImageState = {type: "file", file: action.file, src: URL.createObjectURL(action.file)};
        newState[action.uuid] = newImageState;
        return newState;
      }
      case "SET_EMBEDDING": {
        if (!state.hasOwnProperty(action.uuid)) return state;
        let newState = {...state};
        let newImageState = {...state[action.uuid], embedding: action.embedding};
        newState[action.uuid] = newImageState;
        return newState;
      }
      case "DELETE_IMAGE": {
        let newState = {...state};
        delete newState[action.uuid];
        return newState;
      }
      default:
        throw new Error();
    }
  };

  const [knnImages, knnImagesDispatch] = useReducer(knnReducer, {});
  const [knnUseSpatial, setKnnUseSpatial] = useState(false);

  // Run queries after dataset info has loaded and whenever user clicks "query" button
  const [datasetIncludeTags, setDatasetIncludeTags] = useState([]);
  const [datasetExcludeTags, setDatasetExcludeTags] = useState([]);
  const [googleQuery, setGoogleQuery] = useState("");
  const [orderingMode, setOrderingMode] = useState(orderingModes[0].id);
  const [orderByClusterSize, setOrderByClusterSize] = useState(true);
  const [clusteringStrength, setClusteringStrength] = useState(20);

  const [scoreRange, setScoreRange] = useState([0, 100]);

  const svmPopoverRepositionFunc = useRef();
  const [svmPopoverOpen, setSvmPopoverOpen] = useState(false);
  const [svmAugmentNegs, setSvmAugmentNegs] = useState(true);
  const [svmPosTags, setSvmPosTags] = useState([]);
  const [svmNegTags, setSvmNegTags] = useState([]);
  const [svmAugmentIncludeTags, setSvmAugmentIncludeTags] = useState([]);
  const [svmAugmentExcludeTags, setSvmAugmentExcludeTags] = useState([]);
  const [svmModel, setSvmModel] = useState([]);

  const [rankingModel, setRankingModel] = useState([]);

  const [captionQuery, setCaptionQuery] = useState("");
  const [captionQueryEmbedding, setCaptionQueryEmbedding] = useState("");

  const [subset, setSubset_] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [queryResultData, setQueryResultData] = useState({
    type: null,
    images: [],
    clustering: []
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainedSvmData, setTrainedSvmData] = useState(null);

  const trainSvm = async () => {
    const url = new URL(`${endpoints.trainSvm}/${datasetName}`);
    url.search = new URLSearchParams({
      index_id: datasetInfo.index_id,
      pos_tags: svmPosTags.map(t => `${t.category}:${t.value}`),
      neg_tags: svmNegTags.map(t => `${t.category}:${t.value}`),
      augment_negs: svmAugmentNegs,
      include: svmAugmentIncludeTags.map(t => `${t.category}:${t.value}`),
      exclude: svmAugmentExcludeTags.map(t => `${t.category}:${t.value}`),
    }).toString();
    const svmData = await fetch(url, {
      method: "GET",
    }).then(r => r.json());

    setTrainedSvmData({...svmData, date: Date.now()});
  };
  useEffect(() => {
    if (isTraining) trainSvm().finally(() => setIsTraining(false));
  }, [isTraining]);

  const runQuery = async () => {
    setClusterIsOpen(false);
    setSelection({});

    let url;
    let body = {
      num: 1000,
      index_id: datasetInfo.index_id,
      include: datasetIncludeTags.map(t => `${t.category}:${t.value}`),
      exclude: datasetExcludeTags.map(t => `${t.category}:${t.value}`),
      subset: subset.map(im => im.id),
      score_min: scoreRange[0] / 100,
      score_max: scoreRange[1] / 100,
    };

    if (orderingMode === "id" || orderingMode === "random") {
      url = new URL(`${endpoints.getNextImages}/${datasetName}`);
      body.order = orderingMode;
    } else if (orderingMode === "knn") {
      url = new URL(`${endpoints.queryKnn}/${datasetName}`);
      body.embeddings = Object.values(knnImages).map(i => i.embedding);
      body.use_full_image = !knnUseSpatial;
    } else if (orderingMode === "svm") {
      url = new URL(`${endpoints.querySvm}/${datasetName}`);
      body.svm_vector = trainedSvmData.svm_vector;
      if (svmModel[0]) body.model = svmModel[0].with_output.model_id;
    } else if (orderingMode === "dnn") {
      url = new URL(`${endpoints.queryRanking}/${datasetName}`);
      body.model = rankingModel[0].with_output.model_id;
    } else if (orderingMode === "clip") {
      url = new URL(`${endpoints.queryKnn}/${datasetName}`);
      body.embeddings = [captionQueryEmbedding];
      body.model = "clip";
      body.use_dot_product = true;
    } else {
      console.error(`Query type (${orderingMode}) not implemented`);
      return;
    }
    const results = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(r => r.json());

    const images = results.paths.map((path, i) => {
      let filename = path.substring(path.lastIndexOf("/") + 1);
      let id = filename.substring(0, filename.indexOf("."));
      return {
        name: filename,
        src: path,
        id: results.identifiers[i],
        thumb: `https://storage.googleapis.com/foragerml/thumbnails/${datasetInfo.index_id}/${id}.jpg`,
      };
    });

    if (results.type !== "svm") {
      setTrainedSvmData(null);
    }
    if ((results.type === "knn" || results.type === "svm") &&
        (queryResultData.type !== "knn" && queryResultData.type !== "svm")) {
      setOrderByClusterSize(false);
    }

    setQueryResultData({
      images,
      clustering: results.clustering,
      type: results.type,
    });
  };
  useEffect(() => {
    if (isLoading) runQuery().finally(() => setIsLoading(false));
  }, [isLoading]);

  const setSubset = (subset) => {
    setSubset_(subset);
    setIsLoading(true);
  }

  // Automatically (re-)cluster whenever new results load; also run this manually when
  // the user releases the cluster strength slider
  const [clusters, setClusters] = useState([]);

  const recluster = () => {
    if (clusteringStrength == 0) {
      setClusters(queryResultData.images.map(i => [i]));
    } else {
      const thresh = Math.pow(clusteringStrength / 100, 2);
      let ds = disjointSet();
      for (let image of queryResultData.images) {
        ds.add(image);
      }
      for (let [a, b, dist] of queryResultData.clustering) {
        if (dist > thresh) break;
        ds.union(queryResultData.images[a], queryResultData.images[b]);
      }
      const clusters = ds.extract();
      ds.destroy();
      if (orderByClusterSize) clusters.sort((a, b) => b.length - a.length);
      setClusters(clusters);
    }
  }
  useEffect(recluster, [queryResultData, setClusters, orderByClusterSize]);

  //
  // MODE
  //
  const [mode, setMode_] = useState("explore");
  const [labelModeCategories, setLabelModeCategories_] = useState([]);
  const [dnnAdvancedIsOpen, setDnnAdvancedIsOpen] = useState(false);
  const [modelStatus, setModelStatus] = useState({});

  const [modelInfo, setModelInfo] = useState([]);
  const [dnnType, setDnnType] = useState(dnns[0].id);
  const [dnnAugmentNegs, setDnnAugmentNegs] = useState(true);
  const [dnnPosTags, setDnnPosTags] = useState([]);
  const [dnnNegTags, setDnnNegTags] = useState([]);
  const [requestDnnTraining, setRequestDnnTraining] = useState(false);
  const [clusterId, setClusterId] = useState(null);
  const [clusterCreating, setClusterCreating] = useState(false);
  const [clusterStatus, setClusterStatus] = useState({});
  const [modelId, setModelId] = useState(null);
  const [dnnIsTraining, setDnnIsTraining] = useState(false);
  const [modelEpoch, setModelEpoch] = useState(1);
  const [modelName, setModelName] = useState("");
  const [prevModelId, setPrevModelId] = useState(null);

  const [dnnInferenceModel, setDnnInferenceModel] = useState([]);
  const [requestDnnInference, setRequestDnnInference] = useState(false);
  const [dnnIsInferring, setDnnIsInferring] = useState(false);
  const [dnnInferenceJobId, setDnnInferenceJobId] = useState(null);
  const [dnnInferenceStatus, setDnnInferenceStatus] = useState({});

  useEffect(async () => {
    const getModelsUrl = new URL(`${endpoints.getModels}/${datasetName}`);
    const res = await fetch(getModelsUrl, {
      method: "GET",
      headers: {"Content-Type": "application/json"},
    }).then(res => res.json());
    console.log(res.models);
    setModelInfo(res.models);
  }, [modelEpoch, dnnIsInferring])

  const autofillModelName = () => {
    if (!!!(username)) return;
    const name = username.slice(0, username.indexOf("@"));
    const date = dateFormat(new Date(), "mm-dd-yy_HH-MM");
    setModelName(`${name}_${date}`);
  };

  const setMode = (mode) => {
    setMode_(mode);
    if (mode === "train" && !requestDnnTraining) autofillModelName();
    if (svmPopoverRepositionFunc.current) svmPopoverRepositionFunc.current();
  }

  useEffect(async () => {
    let _clusterId = clusterId;
    if ((requestDnnTraining || requestDnnInference) && !_clusterId) {
      // Start cluster
      const startClusterUrl = new URL(`${endpoints.startCluster}`);
      var clusterResponse = await fetch(startClusterUrl, {
        method: "POST",
      }).then(r => r.json());

      _clusterId = clusterResponse.cluster_id;

      setClusterId(_clusterId);
      setClusterCreating(true);
    }
  }, [requestDnnTraining, requestDnnInference, clusterId]);

  const checkClusterStatus = async () => {
    // Wait until cluster is booted
    const clusterStatusUrl = new URL(`${endpoints.clusterStatus}/${clusterId}`);
    const _clusterStatus = await fetch(clusterStatusUrl, {
      method: "GET",
    }).then(r => r.json());
    setClusterStatus(_clusterStatus);
    if (_clusterStatus.ready) {
      setClusterCreating(false);
    }
  }
  useInterval(checkClusterStatus, clusterCreating ? 3000 : null);

  const startTrainingNewDnn = () => {
    setModelEpoch(0);
    setModelId(null);
    setDnnIsTraining(false);
    setRequestDnnTraining(true);
  };

  const stopTrainingDnn = () => {
    setRequestDnnTraining(false);
    autofillModelName();
  };

  const trainDnnEpoch = async () => {
    let _clusterId = clusterId;
    // Start training DNN
    const url = new URL(`${endpoints.trainModel}/${datasetName}`);
    let body = {
      model_name: modelName,
      cluster_id: _clusterId,
      bucket: "foragerml",
      index_id: datasetInfo.index_id,
      pos_tags: dnnPosTags.map(t => `${t.category}:${t.value}`).join(","),
      neg_tags: dnnNegTags.map(t => `${t.category}:${t.value}`).join(","),
      augment_negs: dnnAugmentNegs.toString(),
      aux_label_type: "imagenet",
    }
    if (modelId) {
      body.resume = modelId;
    }
    const _modelId = await fetch(url, {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        'Content-type': 'application/json; charset=UTF-8'
      }
    }).then(r => r.json()).then(r => r.model_id);
    setModelId(_modelId);
    setDnnIsTraining(true);
  };

  useEffect(() => {
    if (requestDnnTraining && clusterStatus.ready && !dnnIsTraining) {
      trainDnnEpoch();
    }
  }, [requestDnnTraining, clusterStatus, dnnIsTraining]);

  const checkDnnStatus = async () => {
      // Wait until DNN trains for 1 epoch
    const modelStatusUrl = new URL(`${endpoints.modelStatus}/${modelId}`);
    const _modelStatus = await fetch(modelStatusUrl, {
      method: "GET",
    }).then(r => r.json());
    setModelStatus(_modelStatus);
    if (_modelStatus.has_model) {
      setDnnIsTraining(false);
      setPrevModelId(modelId);
      setModelEpoch(modelEpoch + 1);
    }
    if (_modelStatus.failed) {
      setRequestDnnTraining(false);
      setDnnIsTraining(false);
    }
  };
  useInterval(checkDnnStatus, dnnIsTraining ? 3000 : null)

  const startDnnInference = () => {
    setDnnIsInferring(false);
    setRequestDnnInference(true);
  };

  const stopDnnInference = () => {
    const stop = async () => {
      const url = new URL(`${endpoints.stopModelInference}/${dnnInferenceJobId}`);
      const resp = await fetch(url, {
        method: "POST",
        headers: {
          'Content-type': 'application/json; charset=UTF-8'
        }
      }).then(r => r.json());
      setDnnIsInferring(false);
    }
    setRequestDnnInference(false);
    stop();
  };

  const dnnInference = async () => {
    let _clusterId = clusterId;
    // Start training DNN
    const url = new URL(`${endpoints.modelInference}/${datasetName}`);
    let body = {
      model_id: dnnInferenceModel[0].latest.model_id,
      cluster_id: _clusterId,
      bucket: "foragerml",
      index_id: datasetInfo.index_id,
    }
    const _jobId = await fetch(url, {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        'Content-type': 'application/json; charset=UTF-8'
      }
    }).then(r => r.json()).then(r => r.job_id);
    setDnnInferenceJobId(_jobId);
    setDnnIsInferring(true);
  };

  useEffect(() => {
    if (requestDnnInference && clusterStatus.ready && !dnnIsInferring) {
      dnnInference();
    }
  }, [requestDnnInference, clusterStatus, dnnIsInferring, dnnInferenceModel]);

  const checkDnnInferenceStatus = async () => {
    const inferenceStatusUrl = new URL(`${endpoints.modelInferenceStatus}/${dnnInferenceJobId}`);
    const _inferenceStatus = await fetch(inferenceStatusUrl, {
      method: "GET",
    }).then(r => r.json());
    setDnnInferenceStatus(_inferenceStatus);
    if (_inferenceStatus.has_output) {
      setRequestDnnInference(false);
      setDnnIsInferring(false);
    }
  };
  useInterval(checkDnnInferenceStatus, dnnIsInferring ? 3000 : null)

  const setLabelModeCategories = (selection) => {
    if (selection.length === 0) {
      setLabelModeCategories_([]);
    } else {
      let c = selection[selection.length - 1];
      if (c.customOption) {  // new
        c = c.label;

        let newCategories = {...datasetInfo.categories};
        newCategories[c] = [];  // no custom values to start
        setCategories(newCategories);
      }
      setLabelModeCategories_([c]);
    }
  };

  //
  // RENDERING
  //

  return (
    <div className={`main ${isLoading ? "loading" : ""}`}>
      <SignInModal
        isOpen={loginIsOpen}
        toggle={() => setLoginIsOpen(false)}
        loginUsername={loginUsername}
        loginPassword={loginPassword}
        setLoginUsername={setLoginUsername}
        setLoginPassword={setLoginPassword}
        login={login}
      />
      <TagManagementModal
        isOpen={tagManagementIsOpen}
        toggle={toggleTagManagement}
        datasetName={datasetName}
        datasetCategories={datasetInfo.categories}
        setDatasetCategories={setCategories}
        username={username}
        isReadOnly={!!!(username)}
      />
      <ClusterModal
        isOpen={clusterIsOpen}
        setIsOpen={setClusterIsOpen}
        isImageOnly={clusteringStrength == 0}
        isReadOnly={!!!(username)}
        selection={selection}
        setSelection={setSelection}
        clusters={clusters}
        findSimilar={(image) => {
          const uuid = uuidv4();
          knnImagesDispatch({
            type: "ADD_IMAGE_FROM_DATASET",
            image,
            uuid,
          });
          setOrderingMode("knn");
          setClusterIsOpen(false);
          setSelection({});
          generateEmbedding({image_id: image.id}, uuid);
        }}
        tags={datasetInfo.categories}
        setCategories={setCategories}
        username={username}
        setSubset={setSubset}
        mode={mode}
        labelCategory={labelModeCategories[0]}
      />
      <Navbar color="primary" className="text-light justify-content-between" dark expand="sm">
        <Container fluid>
          <span>
            <NavbarBrand href="/"><b>Forager</b></NavbarBrand>
            <NavbarBrand className="font-weight-normal" id="dataset-name">{datasetName}</NavbarBrand>
          </span>
          <span>
            <Nav navbar>
              <NavItem active={mode === "explore"}>
                <NavLink href="#" onClick={(e) => {
                  setMode("explore");
                  e.preventDefault();
                }}>Explore</NavLink>
              </NavItem>
              <NavItem active={mode === "label"}>
                <NavLink href="#" onClick={(e) => {
                  setMode("label");
                  e.preventDefault();
                }}>Label</NavLink>
              </NavItem>
              <NavItem active={mode === "train"}>
                <NavLink href="#" onClick={(e) => {
                  setMode("train");
                  e.preventDefault();
                }}>Train</NavLink>
              </NavItem>
            </Nav>
          </span>
          <div>
            <span className="mr-4" onClick={toggleTagManagement} style={{cursor: "pointer"}}>
              Manage Tags
            </span>
            <span>
              {username ?
                <>{username} (<a href="#" onClick={(e) => {
                  setUsername("");
                  e.preventDefault();
                }}>Sign out</a>)</> :
                <a href="#" onClick={(e) => {
                  setLoginUsername("");
                  setLoginPassword("");
                  setLoginIsOpen(true);
                  e.preventDefault();
                }}>Sign in</a>
              }
            </span>
          </div>
        </Container>
      </Navbar>
      <Popover
        placement="bottom"
        isOpen={popoverOpen}
        target="dataset-name"
        toggle={() => setPopoverOpen(!popoverOpen)}
        trigger="hover focus"
        fade={false}
      >
        <PopoverBody>
          <div><b>Dataset size:</b> {datasetInfo.num_images} image{datasetInfo.num_images === 1 ? "" : "s"}</div>
          <div><b>Index status:</b> {datasetInfo.index_id ? "Created" : "Not created"}</div>
        </PopoverBody>
      </Popover>
      {mode !== "explore" && <div className="border-bottom py-2 mode-container">
        <Container fluid>
            {mode === "label" && <div className="d-flex flex-row align-items-center justify-content-between">
              <Typeahead
                multiple
                id="label-mode-bar"
                className="typeahead-bar mr-2"
                options={sortBy(Object.keys(datasetInfo.categories), c => c.toLowerCase())}
                placeholder="Category to label"
                selected={labelModeCategories}
                onChange={setLabelModeCategories}
                newSelectionPrefix="New category: "
                allowNew={true}
              />
              <div className="text-nowrap">
                {LABEL_VALUES.map(([value, name], i) =>
                  <>
                    <kbd>{(i + 1) % 10}</kbd> <span className={`rbt-token ${value}`}>{name.toLowerCase()}</span>&nbsp;
                  </>)}
                {labelModeCategories.length > 0 &&
                  (datasetInfo.categories[labelModeCategories[0]] || []).map((name, i) =>
                  <>
                    <kbd>{(LABEL_VALUES.length + i + 1) % 10}</kbd> <span className="rbt-token CUSTOM">{name}</span>&nbsp;
                  </>)}
                {labelModeCategories.length > 0 &&
                  <NewModeInput
                    category={labelModeCategories[0]}
                    categories={datasetInfo.categories}
                    setCategories={setCategories}
                  />
                }
              </div>
            </div>}
            {mode === "train" && <>
              <div className="d-flex flex-row align-items-center justify-content-between mb-2">
                {requestDnnTraining ? <>
                  <div className="d-flex flex-row align-items-center">
                    <Spinner color="dark" className="my-1 mr-2" />
                    {clusterStatus.ready ?
                      <span><b>Training</b> model <b>{modelName}</b> (Epoch {modelEpoch}), Time left: {modelStatus.training_time_left && modelStatus.training_time_left >= 0 ? new Date(Math.max(modelStatus.training_time_left, 0) * 1000).toISOString().substr(11, 8) : 'estimating...'} </span> :
                      <span><b>Starting cluster</b></span>
                    }
                  </div>
                  <Button
                    color="danger"
                    onClick={stopTrainingDnn}
                  >Stop training</Button>
                </> : <>
                  <FontAwesomeIcon
                    icon={dnnAdvancedIsOpen ? faChevronUp : faCog}
                    style={{
                      cursor: "pointer",
                      position: "absolute",
                    }}
                    onClick={() => setDnnAdvancedIsOpen(!dnnAdvancedIsOpen)}
                  />
                  <FormGroup className="mb-0 ml-3">
                    <select className="custom-select mr-2" value={dnnType} onChange={e => setDnnType(e.target.value)}>
                      {dnns.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
                    </select>
                    <ReactSVG className="icon" src="assets/arrow-caret.svg" />
                  </FormGroup>
                  <Input
                    className="mr-2"
                    placeholder="Model name"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                  />
                  <CategoryInput
                    id="dnn-pos-bar"
                    className="mr-2"
                    placeholder="Positive example tags"
                    disabled={requestDnnTraining}
                    categories={datasetInfo.categories}
                    setCategories={setCategories}
                    selected={dnnPosTags}
                    setSelected={setDnnPosTags}
                  />
                  <CategoryInput
                    id="dnn-neg-bar"
                    className="mr-2"
                    placeholder="Negative example tags"
                    disabled={requestDnnTraining}
                    categories={datasetInfo.categories}
                    setCategories={setCategories}
                    selected={dnnNegTags}
                    setSelected={setDnnNegTags}
                  />
                  <div className="my-2 custom-control custom-checkbox">
                    <input
                      type="checkbox"
                      className="custom-control-input"
                      id="dnn-augment-negs-checkbox"
                      disabled={requestDnnTraining}
                      checked={dnnAugmentNegs}
                      onChange={(e) => setDnnAugmentNegs(e.target.checked)}
                    />
                    <label className="custom-control-label text-nowrap mr-2" htmlFor="dnn-augment-negs-checkbox">
                      Auto-augment negative set
                    </label>
                  </div>
                  <Button
                    color="light"
                    onClick={startTrainingNewDnn}
                    disabled={!!!(username) || dnnPosTags.length === 0 || (dnnNegTags.length === 0 && !dnnAugmentNegs) || requestDnnTraining}
                    >Start training
                  </Button>
                </>}
              </div>
              <Collapse isOpen={dnnAdvancedIsOpen && !requestDnnTraining} timeout={200} className="pb-2">
                @FAIT ADD ADVANCED SETTINGS HERE
              </Collapse>
              <div className="d-flex flex-row align-items-center justify-content-between mt-2 mb-1">
              {requestDnnInference ? <>
                <div className="d-flex flex-row align-items-center">
                  <Spinner color="dark" className="my-1 mr-2" />
                  {clusterStatus.ready ?
                    <span><b>Inferring</b> model <b>{modelName}</b>, Time left: {dnnInferenceStatus.time_left && dnnInferenceStatus.time_left >= 0 ? new Date(Math.max(dnnInferenceStatus.time_left, 0) * 1000).toISOString().substr(11, 8) : 'estimating...'} </span> :
                    <span><b>Starting cluster</b></span>
                  }
                </div>
                <Button
                  color="danger"
                  onClick={stopDnnInference}
                >Stop inference</Button>
                </> : <>
                <FeatureInput
                  id="dnn-inference-model-bar"
                  className="mr-2"
                  placeholder="Model to perform inference for"
                  features={modelInfo.filter(m => m.latest.has_checkpoint && !m.latest.has_output)}
                  selected={dnnInferenceModel}
                  setSelected={setDnnInferenceModel}
                />
                <Button
                  color="light"
                  onClick={startDnnInference}
                  disabled={!!!(username) || !!!(dnnInferenceModel[0]) || requestDnnInference}
              >Start inference</Button>
                </>}
              </div>
              </>}
            {modelStatus.failed ?
             <div className="d-flex flex-row align-items-center justify-content-between">
              <div className="my-12">{modelStatus.failure_reason}</div>
              </div> : null }
        </Container>
      </div>}
      <div className="app">
        <div className="query-container sticky">
          <Container fluid>
            <div className="d-flex flex-row align-items-center">
              <CategoryInput
                id="dataset-include-bar"
                className="mr-2"
                placeholder="Tags to include"
                categories={datasetInfo.categories}
                setCategories={setCategories}
                selected={datasetIncludeTags}
                setSelected={setDatasetIncludeTags}
              />
              <CategoryInput
                id="dataset-exclude-bar"
                placeholder="Tags to exclude"
                categories={datasetInfo.categories}
                setCategories={setCategories}
                selected={datasetExcludeTags}
                setSelected={setDatasetExcludeTags}
              />
              <FormGroup className="mb-0">
                <select className="custom-select mx-2" id="ordering-mode" value={orderingMode} onChange={e => setOrderingMode(e.target.value)}>
                  {orderingModes.map((m) => <option key={m.id} value={m.id} disabled={m.disabled}>{m.label}</option>)}
                </select>
                <ReactSVG className="icon" src="assets/arrow-caret.svg" />
              </FormGroup>
              <Button
                color="light"
                onClick={() => setIsLoading(true)}
                disabled={
                  (orderingMode === "svm" && !!!(trainedSvmData)) ||
                  (orderingMode === "knn" && (size(knnImages) === 0 || some(Object.values(knnImages).map(i => !(i.embedding))))) ||
                  (orderingMode === "dnn" && !!!(rankingModel[0])) ||
                  (orderingMode === "clip" && !!!(captionQueryEmbedding))
                }
              >Run query</Button>
            </div>
            <div className="mt-2 mb-1 d-flex flex-row-reverse justify-content-between">
              <div className="d-flex flex-row align-items-center">
                <div className="custom-switch custom-control mr-4">
                  <Input type="checkbox" className="custom-control-input"
                    id="order-by-cluster-size-switch"
                    checked={orderByClusterSize}
                    onChange={(e) => setOrderByClusterSize(e.target.checked)}
                  />
                  <label className="custom-control-label text-nowrap" htmlFor="order-by-cluster-size-switch">
                    Order by cluster size
                  </label>
                </div>
                <label className="mb-0 mr-2 text-nowrap">
                  Clustering strength:
                </label>
                <Slider
                  value={clusteringStrength}
                  onChange={setClusteringStrength}
                  onAfterChange={recluster}
                />
              </div>
              {(queryResultData.type === "svm" || queryResultData.type === "ranking") && <div className="d-flex flex-row align-items-center">
                <label className="mb-0 mr-2 text-nowrap">Score range:</label>
                <Range
                  allowCross={false}
                  value={scoreRange}
                  onChange={setScoreRange}
                  onAfterChange={() => setIsLoading(true)}
                />
                <span className="mb-0 ml-2 text-nowrap text-muted">
                  ({Number(scoreRange[0] / 100).toFixed(2)} to {Number(scoreRange[1] / 100).toFixed(2)})
                </span>
              </div>}
              {subset.length > 0 && <div className="rbt-token rbt-token-removeable alert-secondary">
                Limited to {subset.length} image{subset.length !== 1 && "s"}
                <button aria-label="Remove" className="close rbt-close rbt-token-remove-button" type="button" onClick={() => setSubset([])}>
                  <span aria-hidden="true">×</span><span className="sr-only">Remove</span>
                </button>
              </div>}
            </div>
          </Container>
        </div>
        {orderingMode === "svm" && <Popover
          placement="bottom"
          isOpen={!clusterIsOpen && (svmPopoverOpen || isTraining || !!!(trainedSvmData))}
          target="ordering-mode"
          trigger="hover"
          toggle={() => setSvmPopoverOpen(!svmPopoverOpen)}
          fade={false}
          popperClassName={`svm-popover ${isTraining ? "loading" : ""}`}
        >
          {({ scheduleUpdate }) => {
            svmPopoverRepositionFunc.current = scheduleUpdate;
            return (
              <PopoverBody>
                <div>
                  <CategoryInput
                    id="svm-pos-bar"
                    className="mt-1"
                    placeholder="Positive example tags"
                    categories={datasetInfo.categories}
                    setCategories={setCategories}
                    selected={svmPosTags}
                    disabled={isTraining}
                    setSelected={selected => {
                      setSvmPosTags(selected);
                      setTrainedSvmData(null);
                    }}
                  />
                  <CategoryInput
                    id="svm-neg-bar"
                    className="mt-2 mb-3"
                    placeholder="Negative example tags"
                    categories={datasetInfo.categories}
                    setCategories={setCategories}
                    selected={svmNegTags}
                    disabled={isTraining}
                    setSelected={selected => {
                      setSvmNegTags(selected);
                      setTrainedSvmData(null);
                    }}
                  />
                  <FeatureInput
                    id="svm-model-bar"
                    className="mb-2"
                    placeholder="Model features to use (optional)"
                    features={modelInfo.filter(m => m.with_output)}
                    disabled={isTraining}
                    selected={svmModel}
                    setSelected={selected => {
                      setSvmModel(selected);
                      setTrainedSvmData(null);
                    }}
                  />
                  <div className="mt-1 custom-control custom-checkbox">
                    <input
                      type="checkbox"
                      className="custom-control-input"
                      id="svm-augment-negs-checkbox"
                      disabled={isTraining}
                      checked={svmAugmentNegs}
                      onChange={(e) => {
                        setSvmAugmentNegs(e.target.checked);
                        setTrainedSvmData(null);
                      }}
                    />
                    <label className="custom-control-label" htmlFor="svm-augment-negs-checkbox">
                      Auto-augment negative set
                    </label>
                  </div>
                  {svmAugmentNegs && <>
                    <CategoryInput
                      id="svm-augment-negs-include-bar"
                      className="mt-2"
                      placeholder="Tags to include in auto-negative pool"
                      categories={datasetInfo.categories}
                      setCategories={setCategories}
                      selected={svmAugmentIncludeTags}
                      disabled={isTraining}
                      setSelected={selected => {
                        setSvmAugmentIncludeTags(selected);
                        setTrainedSvmData(null);
                      }}
                    />
                    <CategoryInput
                      id="svm-augment-negs-exclude-bar"
                      className="mt-2 mb-1"
                      placeholder="Tags to exclude from auto-negative pool"
                      categories={datasetInfo.categories}
                      setCategories={setCategories}
                      selected={svmAugmentExcludeTags}
                      disabled={isTraining}
                      setSelected={selected => {
                        setSvmAugmentExcludeTags(selected);
                        setTrainedSvmData(null);
                      }}
                    />
                  </>}
                  <Button
                    color="light"
                    onClick={() => setIsTraining(true)}
                    disabled={svmPosTags.length === 0 || (svmNegTags.length === 0 && !svmAugmentNegs) || isTraining}
                    className="mt-2 mb-1 w-100"
                  >Train</Button>
                  {!!(trainedSvmData) && <div className="mt-1">
                    Trained model ({trainedSvmData.num_positives} positives,{" "}
                    {trainedSvmData.num_negatives} negatives) &mdash;{" "}
                    precision {Number(trainedSvmData.precision).toFixed(2)},
                    recall {Number(trainedSvmData.recall).toFixed(2)},
                    F1 {Number(trainedSvmData.f1).toFixed(2)}){" "}
                    <ReactTimeAgo date={trainedSvmData.date} timeStyle="mini"/> ago
                  </div>}
                </div>
              </PopoverBody>
            );
          }}
        </Popover>}
        {orderingMode === "knn" && <KnnPopover
          images={knnImages}
          dispatch={knnImagesDispatch}
          generateEmbedding={generateEmbedding}
          useSpatial={knnUseSpatial}
          setUseSpatial={setKnnUseSpatial}
          hasDrag={hasDrag}
          canBeOpen={!clusterIsOpen}
        />}
        {orderingMode === "dnn" && <ModelRankingPopover
          features={modelInfo.filter(m => m.with_output)}
          rankingModel={rankingModel}
          setRankingModel={setRankingModel}
          canBeOpen={!clusterIsOpen}
        />}
        {orderingMode === "clip" && <CaptionSearchPopover
          text={captionQuery}
          setText={setCaptionQuery}
          textEmbedding={captionQueryEmbedding}
          setTextEmbedding={setCaptionQueryEmbedding}
          canBeOpen={!clusterIsOpen}
        />}
        <Container fluid>
          {(!!!(datasetInfo.isNotLoaded) && !isLoading && queryResultData.images.length == 0) &&
            <p className="text-center text-muted">No results match your query.</p>}
          <Row>
            <Col className="stack-grid">
              {clusters.map((images, i) =>
                <ImageStack
                  id={i}
                  onClick={() => {
                    setSelection({cluster: i});
                    setClusterIsOpen(true);
                  }}
                  images={images}
                  showLabel={clusteringStrength > 0}
                  key={i}
                />
              )}
            </Col>
          </Row>
        </Container>
      </div>
    </div>
  );
}

export default App;
