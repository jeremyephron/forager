import React, { useState, useEffect, useCallback } from "react";
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
} from "reactstrap";
import { Typeahead } from "react-bootstrap-typeahead";
import { ReactSVG } from "react-svg";
import Slider, { Range } from "rc-slider";
import Emoji from "react-emoji-render";
import ReactTimeAgo from "react-time-ago";

import fromPairs from "lodash/fromPairs";
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
} from "./components";

var disjointSet = require("disjoint-set");

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
  {id: "knn", label: "KNN", disabled: true},
];

const dnns = [
  {id: "dnn", label: "DNN w/ BG Splitting"},
];

const endpoints = fromPairs(toPairs({
  getDatasetInfo: "get_dataset_info_v2",
  getNextImages: "get_next_images_v2",
  trainSvm: "train_svm_v2",
  querySvm: "query_svm_v2",
  queryKnn: "query_knn_v2",
  trainModel: "train_model_v2",
  modelStatus: "model_v2",
  startCluster: "start_cluster",
  clusterStatus: "cluster",
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const App = () => {
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

  useEffect(async () => {
    const url = new URL(`${endpoints.getDatasetInfo}/${datasetName}`);
    setDatasetInfo(await fetch(url, {
      method: "GET",
    }).then(r => r.json()));
    setIsLoading(true);
  }, [datasetName]);

  const setTags = (tags) => setDatasetInfo({...datasetInfo, categories: tags});

  // Run queries after dataset info has loaded and whenever user clicks "query" button
  const [datasetIncludeTags, setDatasetIncludeTags] = useState([]);
  const [datasetExcludeTags, setDatasetExcludeTags] = useState([]);
  const [googleQuery, setGoogleQuery] = useState("");
  const [orderingMode, setOrderingMode] = useState(orderingModes[0].id);
  const [orderByClusterSize, setOrderByClusterSize] = useState(true);
  const [clusteringStrength, setClusteringStrength] = useState(20);
  const [orderingModePopoverOpen, setOrderingModePopoverOpen] = useState(false);
  const [svmScoreRange, setSvmScoreRange] = useState([0, 100]);
  const [svmAugmentNegs, setSvmAugmentNegs] = useState(true);
  const [svmPosTags, setSvmPosTags] = useState([]);
  const [svmNegTags, setSvmNegTags] = useState([]);

  const [knnImage, setKnnImage] = useState({});
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
    let params = {
      num: 1000,
      index_id: datasetInfo.index_id,
      include: datasetIncludeTags.map(t => `${t.category}:${t.value}`),
      exclude: datasetExcludeTags.map(t => `${t.category}:${t.value}`),
      subset: subset.map(im => im.id),
    };

    if (orderingMode === "id" || orderingMode === "random") {
      url = new URL(`${endpoints.getNextImages}/${datasetName}`);
      url.search = new URLSearchParams({...params, order: orderingMode}).toString();
    } else if (orderingMode === "knn") {
      url = new URL(`${endpoints.queryKnn}/${datasetName}`);
      url.search = new URLSearchParams({...params,
        image_ids: [knnImage.id]
      }).toString();
    } else if (orderingMode === "svm") {
      url = new URL(`${endpoints.querySvm}/${datasetName}`);
      url.search = new URLSearchParams({...params,
        svm_vector: trainedSvmData.svm_vector,
        score_min: svmScoreRange[0] / 100,
        score_max: svmScoreRange[1] / 100,
      }).toString();
    } else {
      console.error(`Query type (${orderingMode}) not implemented`);
      return;
    }
    const results = await fetch(url, {
      method: "GET",
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
      // Reset all SVM parameters
      setSvmScoreRange([0, 100]);
      setSvmAugmentNegs(true);
      setSvmPosTags([]);
      setSvmNegTags([]);
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

  // Run KNN queries whenever user clicks "find similar" button
  const findSimilar = (image) => {
    setKnnImage(image);
    setOrderingMode("knn");
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
  const [mode, setMode] = useState("explore");
  const [labelModeCategories, setLabelModeCategories_] = useState([]);
  const [modelStatus, setModelStatus] = useState({});
  const [dnnType, setDnnType] = useState(dnns[0].id);
  const [dnnAugmentNegs, setDnnAugmentNegs] = useState(true);
  const [dnnPosTags, setDnnPosTags] = useState([]);
  const [dnnNegTags, setDnnNegTags] = useState([]);
  const [dnnIsTraining, setDnnIsTraining] = useState(false);
  const [clusterStatus, setClusterStatus] = useState({});
  const [clusterId, setClusterId] = useState(null);
  const [modelId, setModelId] = useState(null);
  const [modelEpoch, setModelEpoch] = useState(1);
  const [statusIntervalId, setStatusIntervalId] = useState();
  const [modelName, setModelName] = "kayvonf_04-06-2021";

  const trainDnn = async () => {
    let _clusterId = clusterId;

    if (!_clusterId) {
      // Start cluster
      const startClusterUrl = new URL(`${endpoints.startCluster}`);
      var clusterResponse = await fetch(startClusterUrl, {
        method: "POST",
      }).then(r => r.json());
      _clusterId = clusterResponse.cluster_id;
      setClusterId(_clusterId);

      // Wait until cluster is booted
      while (true) {
        const clusterStatusUrl = new URL(`${endpoints.clusterStatus}/${_clusterId}`);
        const _clusterStatus = await fetch(clusterStatusUrl, {
          method: "GET",
        }).then(r => r.json());
        setClusterStatus(_clusterStatus);
        if (_clusterStatus.ready) break;
        await new Promise(r => setTimeout(r, 3000));  // 3 seconds
      }
    }

    // Start training DNN
    while (true) {
      const url = new URL(`${endpoints.trainModel}/${datasetName}`);
      console.log(dnnAugmentNegs);
      const body = {
        model_name: "TEST_MODEL",
        cluster_id: _clusterId,
        bucket: "foragerml",
        index_id: datasetInfo.index_id,
        pos_tags: dnnPosTags.map(t => `${t.category}:${t.value}`).join(","),
        neg_tags: dnnNegTags.map(t => `${t.category}:${t.value}`).join(","),
        augment_negs: dnnAugmentNegs == "on" ? 'true' : 'false',
        aux_label_type: "imagenet",
      }
      const _modelId = await fetch(url, {
        method: "POST",
        body: JSON.stringify(body),
        headers: {
          'Content-type': 'application/json; charset=UTF-8'
        }
      }).then(r => r.json()).then(r => r.model_id);

      // Wait until DNN trains for 1 epoch
      while (true) {
        const modelStatusUrl = new URL(`${endpoints.modelStatus}/${_clusterId}`);
        const _modelStatus = await fetch(modelStatusUrl, {
          method: "GET",
        }).then(r => r.json());
        setModelStatus(_modelStatus);
        if (_modelStatus.has_model) break;
        await new Promise(r => setTimeout(r, 3000));  // 3 seconds
      }

      setModelEpoch(modelEpoch + 1);
    }
  };

  useEffect(() => {
    if (dnnIsTraining) trainDnn();
  }, [dnnIsTraining]);

  const setLabelModeCategories = (selection) => {
    if (selection.length === 0) {
      setLabelModeCategories_([]);
    } else {
      let c = selection[selection.length - 1];
      if (c.customOption) {  // new
        c = c.label;
        let newCategories = [...datasetInfo.categories, c];
        setDatasetInfo({...datasetInfo, categories: newCategories.sort()});
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
        datasetInfo={datasetInfo}
        setDatasetInfo={setDatasetInfo}
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
        findSimilar={findSimilar}
        tags={datasetInfo.categories}
        setTags={setTags}
        username={username}
        setSubset={setSubset}
        mode={mode}
        labelCategory={labelModeCategories[0]}
      />
      <Navbar color="primary-3" className="text-light justify-content-between" dark expand="sm">
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
          <div className="d-flex flex-row align-items-center justify-content-between">
            {mode === "label" && <>
              <Typeahead
                multiple
                id="label-mode-bar"
                className="typeahead-bar mr-2"
                options={datasetInfo.categories}
                placeholder="Category to label"
                selected={labelModeCategories}
                onChange={setLabelModeCategories}
                newSelectionPrefix="New category: "
                allowNew={true}
              />
              <span className="text-nowrap">
                <b>Key bindings:</b> &nbsp;
                {LABEL_VALUES.map(([value, name], i) =>
                  <>
                    <kbd>{i + 1}</kbd> <span className={`rbt-token ${value}`}>{name.toLowerCase()}</span>
                    {i < LABEL_VALUES.length - 1 && ", "}
                  </>)}
              </span>
            </>}
            {mode === "train" && (dnnIsTraining ? <>
              <div className="d-flex flex-row align-items-center">
                <Spinner color="dark" className="my-1 mr-2" />
                {clusterStatus.ready ?
                  <span><b>Training</b> model <b>{modelName}</b> (Epoch {modelEpoch})</span> :
                  <span><b>Starting cluster</b></span>
                }
              </div>
              <Button
                color="danger"
                onClick={() => setDnnIsTraining(false)}
                disabled={true}
              >Stop training</Button>
            </> : <>
              <FormGroup className="mb-0">
                <select className="custom-select mr-2" value={dnnType} onChange={e => setDnnType(e.target.value)}>
                  {dnns.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
                </select>
                <ReactSVG className="icon" src="assets/arrow-caret.svg" />
              </FormGroup>
              <CategoryInput
                id="dnn-pos-bar"
                className="mr-2"
                placeholder="Positive example tags"
                disabled={dnnIsTraining}
                categories={datasetInfo.categories}
                setCategories={setTags}
                selected={dnnPosTags}
                setSelected={setDnnPosTags}
              />
              <CategoryInput
                id="dnn-neg-bar"
                className="mr-2"
                placeholder="Negative example tags"
                disabled={dnnIsTraining}
                categories={datasetInfo.categories}
                setCategories={setTags}
                selected={dnnNegTags}
                setSelected={setDnnNegTags}
              />
              <div className="my-2 custom-control custom-checkbox">
                <input
                  type="checkbox"
                  className="custom-control-input"
                  id="svm-augment-negs-checkbox"
                  disabled={dnnIsTraining}
                  checked={dnnAugmentNegs}
                  onChange={(e) => setDnnAugmentNegs(e.target.checked)}
                />
                <label className="custom-control-label text-nowrap mr-2" htmlFor="svm-augment-negs-checkbox">
                  Automatically augment negative set
                </label>
              </div>
              <Button
                color="light"
                onClick={() => setDnnIsTraining(true)}
                disabled={dnnPosTags.length === 0 || (dnnNegTags.length === 0 && !dnnAugmentNegs) || dnnIsTraining}
              >Start training</Button>
            </>)}
          </div>
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
                setCategories={setTags}
                selected={datasetIncludeTags}
                setSelected={setDatasetIncludeTags}
              />
              <CategoryInput
                id="dataset-exclude-bar"
                placeholder="Tags to exclude"
                categories={datasetInfo.categories}
                setCategories={setTags}
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
                disabled={orderingMode === "svm" && !!!(trainedSvmData)}
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
              {queryResultData.type === "svm" && <div className="d-flex flex-row align-items-center">
                <label className="mb-0 mr-2 text-nowrap">SVM score range:</label>
                <Range
                  allowCross={false}
                  value={svmScoreRange}
                  onChange={setSvmScoreRange}
                  onAfterChange={() => setIsLoading(true)}
                />
                <span className="mb-0 ml-2 text-nowrap text-muted">
                  ({Number(svmScoreRange[0] / 100).toFixed(2)} to {Number(svmScoreRange[1] / 100).toFixed(2)})
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
          isOpen={orderingModePopoverOpen || isTraining || !!!(trainedSvmData)}
          target="ordering-mode"
          trigger="hover"
          toggle={() => setOrderingModePopoverOpen(!orderingModePopoverOpen)}
          fade={false}
          popperClassName={`svm-popover ${isTraining ? "loading" : ""}`}
        >
          <PopoverBody>
            <div>
              <CategoryInput
                id="svm-pos-bar"
                className="mt-1"
                placeholder="Positive example tags"
                categories={datasetInfo.categories}
                setCategories={setTags}
                selected={svmPosTags}
                setSelected={setSvmPosTags}
                disabled={isTraining}
                onChange={selected => {
                  setSvmPosTags(selected);
                  setTrainedSvmData(null);
                }}
              />
              <CategoryInput
                id="svm-neg-bar"
                className="mt-2 mb-1"
                placeholder="Negative example tags"
                categories={datasetInfo.categories}
                setCategories={setTags}
                selected={svmNegTags}
                setSelected={setSvmNegTags}
                disabled={isTraining}
                onChange={selected => {
                  setSvmNegTags(selected);
                  setTrainedSvmData(null);
                }}
              />

              <div className="my-2 custom-control custom-checkbox">
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
                  Automatically augment negative set
                </label>
              </div>
              <Button
                color="light"
                onClick={() => setIsTraining(true)}
                disabled={svmPosTags.length === 0 || (svmNegTags.length === 0 && !svmAugmentNegs) || isTraining}
                className="mb-1 w-100"
              >Train</Button>
              {!!(trainedSvmData) && <div className="mt-1">
                Trained model ({trainedSvmData.num_positives} positives,{" "}
                {trainedSvmData.num_negatives} negatives &mdash;{" "})
                precision {Number(trainedSvmData.precision).toFixed(2)},
                recall {Number(trainedSvmData.recall).toFixed(2)},
                F1 {Number(2 * trainedSvmData.f1).toFixed(2)}){" "}
                <ReactTimeAgo date={trainedSvmData.date} timeStyle="mini"/> ago
              </div>}
            </div>
          </PopoverBody>
        </Popover>}
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
