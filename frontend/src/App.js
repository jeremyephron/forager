import React, { useState, useEffect, useCallback } from "react";
import {
  Container,
  Row,
  Col,
  Button,
  Nav,
  Navbar,
  NavbarBrand,
  Form,
  FormGroup,
  Input,
  Modal,
  ModalBody,
  Popover,
  PopoverBody,
} from "reactstrap";
import { Typeahead } from "react-bootstrap-typeahead";
import { ReactSVG } from "react-svg";
import Slider, { Range } from "rc-slider";
import Emoji from "react-emoji-render";
import TimeAgo from "javascript-time-ago";
import ReactTimeAgo from "react-time-ago";
import en from "javascript-time-ago/locale/en";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import "react-bootstrap-typeahead/css/Typeahead.css";
import "rc-slider/assets/index.css";
import "./scss/theme.scss";

import {
  ClusterModal,
  ImageStack,
  SignInModal,
  TagManagementModal
} from "./components";

TimeAgo.addDefaultLocale(en)

var disjointSet = require("disjoint-set");

const sources = [
  {id: "dataset", label: "Dataset"},
  // {id: "google", label: "Google"},
]

const orderingModes = [
  {id: "random", label: "Random order"},
  {id: "id", label: "Dataset order"},
  {id: "svm", label: "SVM"},
  {id: "knn", label: "KNN", disabled: true},
]

const endpoints = fromPairs(toPairs({
  getDatasetInfo: "get_dataset_info_v2",
  getNextImages: "get_next_images_v2",
  trainSvm: "train_svm_v2",
  querySvm: "query_svm_v2",
  queryKnn: "query_knn_v2",
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

  // Run queries after dataset info has loaded and whenever user clicks "query" button
  const [source, setSource] = useState(sources[0].id);
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
      pos_tags: svmPosTags,
      neg_tags: svmNegTags,
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
      include: datasetIncludeTags,
      exclude: datasetExcludeTags,
      subset: subset.map(im => im.id),
    };

    if (source === "dataset" && (orderingMode === "id" || orderingMode === "random")) {
      url = new URL(`${endpoints.getNextImages}/${datasetName}`);
      url.search = new URLSearchParams({...params, order: orderingMode}).toString();
    } else if (source === "dataset" && orderingMode === "knn") {
      url = new URL(`${endpoints.queryKnn}/${datasetName}`);
      url.search = new URLSearchParams({...params,
        image_ids: [knnImage.id]
      }).toString();
    } else if (source === "dataset" && orderingMode === "svm") {
      url = new URL(`${endpoints.querySvm}/${datasetName}`);
      url.search = new URLSearchParams({...params,
        svm_vector: trainedSvmData.svm_vector,
        score_min: svmScoreRange[0] / 100,
        score_max: svmScoreRange[1] / 100,
      }).toString();
    } else {
      console.error(`Query type (${source}, ${orderingMode}) not implemented`);
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
        setTags={(tags) => setDatasetInfo({...datasetInfo, categories: tags})}
        username={username}
        setSubset={setSubset}
      />
      <Navbar color="primary" className="text-light mb-2" dark>
        <Container fluid>
          <span>
            <NavbarBrand href="/">Forager</NavbarBrand>
            <NavbarBrand className="font-weight-normal" id="dataset-name">{datasetName}</NavbarBrand>
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
      <div className="app">
        <div className="query-container sticky">
          <Container fluid>
            <Form className="d-flex flex-row align-items-center">
              <FormGroup className="mb-0">
                <select className="custom-select mr-2" onChange={e => setSource(e.target.value)}>
                  {sources.map((s) => <option value={s.id} selected={source === s.id}>{s.label}</option>)}
                </select>
                <ReactSVG className="icon" src="assets/arrow-caret.svg" />
              </FormGroup>
              {(() => {
                if (source === "dataset") {
                  return (
                    <>
                      <Typeahead
                        multiple
                        id="dataset-include-bar"
                        className="typeahead-bar mr-2"
                        placeholder="Tags to include"
                        options={datasetInfo.categories}
                        selected={datasetIncludeTags}
                        onChange={selected => setDatasetIncludeTags(selected)}
                      />
                      <Typeahead
                        multiple
                        id="dataset-exclude-bar"
                        className="typeahead-bar rbt-red"
                        placeholder="Tags to exclude"
                        options={datasetInfo.categories}
                        selected={datasetExcludeTags}
                        onChange={selected => setDatasetExcludeTags(selected)}
                      />
                    </>)
                } else if (source === "google") {
                  return (
                    <Input
                      type="text"
                      placeholder="Query"
                      value={googleQuery}
                      onChange={e => setGoogleQuery(e.target.value)}
                    />
                  )
                }
              })()}
              <FormGroup className="mb-0">
                <select className="custom-select mx-2" id="ordering-mode" onChange={e => setOrderingMode(e.target.value)}>
                  {orderingModes.map((m) => <option value={m.id} selected={orderingMode === m.id} disabled={m.disabled}>{m.label}</option>)}
                </select>
                <ReactSVG className="icon" src="assets/arrow-caret.svg" />
              </FormGroup>
              <Button
                color="primary"
                onClick={() => setIsLoading(true)}
                disabled={orderingMode === "svm" && !!!(trainedSvmData)}
              >Run query</Button>
            </Form>
            <Form className="mt-2 mb-1 d-flex flex-row-reverse justify-content-between">
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
                <label className="mb-0 mr-1 text-nowrap">SVM score range:</label>
                <span className="mb-0 mr-2 text-nowrap text-muted text-monospace text-small">
                  {Number(svmScoreRange[0] / 100).toFixed(2)}
                </span>
                <Range
                  allowCross={false}
                  value={svmScoreRange}
                  onChange={setSvmScoreRange}
                  onAfterChange={() => setIsLoading(true)}
                />
                <span className="mb-0 ml-2 text-nowrap text-muted text-monospace text-small">
                  {Number(svmScoreRange[1] / 100).toFixed(2)}
                </span>
              </div>}
              {subset.length > 0 && <div className="rbt-token rbt-token-removeable alert-secondary">
                Limited to {subset.length} image{subset.length !== 1 && "s"}
                <button aria-label="Remove" className="close rbt-close rbt-token-remove-button" type="button" onClick={() => setSubset([])}>
                  <span aria-hidden="true">×</span><span className="sr-only">Remove</span>
                </button>
              </div>}
            </Form>
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
            <Form>
              <Typeahead
                multiple
                id="svm-pos-bar"
                className="typeahead-bar mt-1"
                placeholder="Positive example tags"
                options={datasetInfo.categories}
                selected={svmPosTags}
                disabled={isTraining}
                onChange={selected => {
                  setSvmPosTags(selected);
                  setTrainedSvmData(null);
                }}
              />
              <Typeahead
                multiple
                id="svm-neg-bar"
                className="typeahead-bar rbt-red mt-2 mb-1"
                placeholder="Negative example tags"
                options={datasetInfo.categories}
                selected={svmNegTags}
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
                  disabled={svmNegTags.length === 0 || isTraining}
                  checked={svmAugmentNegs || svmNegTags.length === 0}
                  onChange={(e) => {
                    setSvmAugmentNegs(e.target.checked);
                    setTrainedSvmData(null);
                  }}
                />
                <label className="custom-control-label" htmlFor="svm-augment-negs-checkbox">
                  Automatically augment negative set with random examples (
                  {svmNegTags.length === 0 ? "required if no explicit negative examples" : "recommended"})
                </label>
              </div>
              <Button
                color="light"
                onClick={() => setIsTraining(true)}
                disabled={svmPosTags.length === 0}
                className="mb-1 w-100"
              >Train</Button>
              {!!(trainedSvmData) && <div className="mt-1">
                <Emoji text=":white_check_mark:" /> Trained model (accuracy{" "}
                {Number(trainedSvmData.accuracy).toFixed(2)}){" "}
                <ReactTimeAgo date={trainedSvmData.date} timeStyle="mini"/> ago
              </div>}
            </Form>
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
