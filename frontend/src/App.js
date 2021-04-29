import React, { useState, useEffect,  useReducer, useRef } from "react";
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
  Form,
  FormGroup,
  Label,
  Input,
  CustomInput,
  InputGroup,
  Modal,
  ModalBody,
  Popover,
  PopoverBody,
  Spinner,
} from "reactstrap";
import { ReactSVG } from "react-svg";
import Slider, { Range } from "rc-slider";
import Emoji from "react-emoji-render";
import ReactTimeAgo from "react-time-ago";
import { v4 as uuidv4 } from "uuid";
import { faTags } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import ReactPaginate from "react-paginate";

import fromPairs from "lodash/fromPairs";
import size from "lodash/size";
import some from "lodash/some";
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
  BulkTagModal,
  ValidatePanel,
  LabelPanel,
  TrainPanel,
} from "./components";

var disjointSet = require("disjoint-set");

const PAGE_SIZE = 1000;

const splits = [
  {id: "train", label: "Training set"},
  {id: "val", label: "Validation set"},
];

const orderingModes = [
  {id: "random", label: "Random order"},
  {id: "id", label: "Dataset order"},
  {id: "svm", label: "SVM"},
  {id: "knn", label: "KNN"},
  {id: "dnn", label: "Model ranking"},
  {id: "clip", label: "Caption search"},
];

const modes = [
  {id: "explore", label: "Explore"},
  {id: "label", label: "Label"},
  {id: "train", label: "Train"},
  {id: "validate", label: "Validate"},
];

const endpoints = fromPairs(toPairs({
  getDatasetInfo: "get_dataset_info_v2",
  getResults: "get_results_v2",
  trainSvm: "train_svm_v2",
  queryImages: "query_images_v2",
  querySvm: "query_svm_v2",
  queryKnn: "query_knn_v2",
  queryRanking: "query_ranking_v2",
  generateEmbedding: 'generate_embedding_v2',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

function MainHeader(props) {
  const [loginIsOpen, setLoginIsOpen] = useState(false);
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  let username = props.username;
  let setUsername = props.setUsername;

  const login = (e) => {
    if (loginUsername !== undefined && loginPassword === "forager") setUsername(loginUsername.trim());
    setLoginIsOpen(false);
    e.preventDefault();
  }


  //
  // TAG MANAGEMENT MODAL
  //
  const [tagManagementIsOpen, setTagManagementIsOpen] = useState(false);
  const toggleTagManagement = () => setTagManagementIsOpen(!tagManagementIsOpen);

  const [popoverOpen, setPopoverOpen] = useState(false);

  let datasetName = props.datasetName;
  let datasetInfo = props.datasetInfo;
  let setCategories = props.setCategories;

  let clusterIsOpen = props.clusterIsOpen;
  let setClusterIsOpen = props.setClusterIsOpen;
  let clusteringStrength = props.clusteringStrength;
  let selection = props.selection;
  let setSelection = props.setSelection;
  let clusters = props.clusters;
  let knnImagesDispatch = props.knnImagesDispatch;
  let setOrderingMode = props.setOrderingMode;
  let generateEmbedding = props.generateEmbedding;
  let setSubset = props.setSubset;
  let mode = props.mode;
  let setMode = props.setMode;
  let labelModeCategory = props.labelModeCategory;

  return (<>
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
    <BulkTagModal
      isOpen={props.bulkTagModalIsOpen}
      toggle={props.toggleBulkTag}
      resultSet={props.queryResultSet}
      categories={datasetInfo.categories}
      setCategories={setCategories}
      username={username}
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
      labelCategory={labelModeCategory}
    />
    <Navbar color="primary" className="text-light justify-content-between" dark expand="sm">
      <Container fluid>
        <span>
          <NavbarBrand href="/"><b>Forager</b></NavbarBrand>
          <NavbarBrand className="font-weight-normal" id="dataset-name">{datasetName}</NavbarBrand>
        </span>
        <span>
          <Nav navbar>
            {modes.map(({ id, label }) => <NavItem active={mode === id}>
              <NavLink href="#" onClick={(e) => {
                setMode(id);
                e.preventDefault();
              }}>{label}</NavLink>
            </NavItem>)}
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
        <div><b>Training set:</b> {datasetInfo.num_train} image{datasetInfo.num_train === 1 ? "" : "s"}</div>
        <div><b>Validation set:</b> {datasetInfo.num_val} image{datasetInfo.num_val === 1 ? "" : "s"}</div>
        <div><b>Index status:</b> {datasetInfo.index_id ? "Created" : "Not created"}</div>
      </PopoverBody>
    </Popover>
  </>);
};


function ClusteringControls(props) {
  let orderByClusterSize = props.orderByClusterSize;
  let setOrderByClusterSize = props.setOrderByClusterSize;
  let clusteringStrength = props.clusteringStrength;
  let setClusteringStrength = props.setClusteringStrength;
  let recluster = props.recluster;
  let username = props.username;
  return (
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
      <Button
        color="light"
        size="sm"
        className="ml-4"
        onClick={props.toggleBulkTag}
        disabled={!!!(username)}
      >
        <FontAwesomeIcon
          icon={faTags}
          className="mr-1"
        />
        Bulk tag results
      </Button>
    </div>);
};


function QueryBar(p) {
  let props = p;
  let datasetInfo = props.datasetInfo;
  let setCategories = props.setCategories;
  let split = props.split;
  let setSplit = props.setSplit;
  let datasetIncludeTags = props.datasetIncludeTags;
  let setDatasetIncludeTags = props.setDatasetIncludeTags;
  let datasetExcludeTags = props.datasetExcludeTags;
  let setDatasetExcludeTags = props.setDatasetExcludeTags;
  let setIsLoading = p.setIsLoading;
  let orderingMode = p.orderingMode;
  let setOrderingMode = props.setOrderingMode;
  let orderingModes = props.orderingModes;
  let trainedSvmData = props.trainedSvmData;
  let knnImages = props.knnImages;
  let rankingModel = props.rankingModel;
  let captionQueryEmbedding = props.captionQueryEmbedding;
  let queryResultSet = props.queryResultSet;
  let scoreRange = props.scoreRange;
  let setScoreRange = props.setScoreRange;
  let subset = props.subset;
  let setSubset = props.setSubset;

  let orderByClusterSize = props.orderByClusterSize;
  let setOrderByClusterSize = props.setOrderByClusterSize;
  let clusteringStrength = props.clusteringStrength;
  let setClusteringStrength = props.setClusteringStrength;
  let recluster = props.recluster;
  return (
    <div className="query-container sticky">
      <Container fluid>
        <div className="d-flex flex-row align-items-center">
          <FormGroup className="mb-0">
            <select className="custom-select mr-2" id="split" value={split} onChange={e => setSplit(e.target.value)}>
              {splits.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
            <ReactSVG className="icon" src="assets/arrow-caret.svg" />
          </FormGroup>
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
            color="primary"
            onClick={() => setIsLoading(true)}
            disabled={
            (orderingMode === "svm" && !!!(trainedSvmData)) ||
            (orderingMode === "knn" && (size(knnImages) === 0 || some(Object.values(knnImages).map(i => !(i.embedding))))) ||
            (orderingMode === "dnn" && !!!(rankingModel)) ||
            (orderingMode === "clip" && !!!(captionQueryEmbedding))
            }
          >Run query</Button>
        </div>
        <div className="mt-2 mb-1 d-flex flex-row-reverse justify-content-between">
          <ClusteringControls
            orderByClusterSize={props.orderByClusterSize}
            setOrderByClusterSize={props.setOrderByClusterSize}
            clusteringStrength={props.clusteringStrength}
            setClusteringStrength={props.setClusteringStrength}
            recluster={props.recluster}
            toggleBulkTag={props.toggleBulkTag}
            username={props.username}
          />
          {(queryResultSet.type === "svm" || queryResultSet.type === "ranking") && <div className="d-flex flex-row align-items-center">
            <label className="mb-0 mr-2 text-nowrap">Score range:</label>
            <Range
              allowCross={false}
              value={scoreRange}
              onChange={setScoreRange}
              onAfterChange={() => p.setIsLoading(true)}
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
    </div>);
};

function OrderingModeSelector(p) {
  let datasetInfo = p.datasetInfo;
  let setCategories = p.setCategories;
  let svmPopoverRepositionFunc = p.svmPopoverRepositionFunc;
  let svmPosTags = p.svmPosTags;
  let setSvmPosTags = p.setSvmPosTags;
  let svmNegTags = p.svmNegTags;
  let setSvmNegTags = p.setSvmNegTags;
  let trainedSvmData = p.trainedSvmData;
  let setTrainedSvmData = p.setTrainedSvmData;
  let svmIsTraining = p.svmIsTraining;
  let svmModel = p.svmModel
  let setSvmModel = p.setSvmModel
  let modelInfo = p.modelInfo;
  let svmAugmentNegs = p.svmAugmentNegs;
  let setSvmAugmentNegs = p.setSvmAugmentNegs;
  let svmAugmentIncludeTags = p.svmAugmentIncludeTags;
  let setSvmAugmentIncludeTags = p.setSvmAugmentIncludeTags;
  let svmAugmentExcludeTags = p.svmAugmentExcludeTags;
  let setSvmAugmentExcludeTags = p.setSvmAugmentExcludeTags;
  let orderingMode = p.orderingMode;
  let clusterIsOpen = p.clusterIsOpen;
  let svmPopoverOpen = p.svmPopoverOpen;

  let knnImages = p.knnImages;
  let knnImagesDispatch = p.knnImagesDispatch;
  let generateEmbedding = p.generateEmbedding;
  let knnUseSpatial = p.knnUseSpatial;
  let setKnnUseSpatial = p.setKnnUseSpatial;
  let hasDrag = p.hasDrag;

  let rankingModel = p.rankingModel;
  let setRankingModel = p.setRankingModel;

  let captionQuery = p.captionQuery;
  let setCaptionQuery = p.setCaptionQuery;
  let captionQueryEmbedding = p.captionQueryEmbedding;
  let setCaptionQueryEmbedding = p.setCaptionQueryEmbedding;

  const svmPopoverBody = ({ scheduleUpdate }) => {
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
            disabled={svmIsTraining}
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
            disabled={svmIsTraining}
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
            disabled={svmIsTraining}
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
              disabled={svmIsTraining}
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
              disabled={svmIsTraining}
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
              disabled={svmIsTraining}
              setSelected={selected => {
                setSvmAugmentExcludeTags(selected);
                setTrainedSvmData(null);
              }}
            />
          </>}
          <Button
            color="light"
            onClick={() => p.setSvmIsTraining(true)}
            disabled={svmPosTags.length === 0 ||
                      (svmNegTags.length === 0 && !svmAugmentNegs) ||
                      svmIsTraining}
            className="mt-2 mb-1 w-100"
          >Train</Button>
          {!!(trainedSvmData) && <div className="mt-1">
            Trained model ({trainedSvmData.num_positives} positives,{" "}
            {trainedSvmData.num_negatives} negatives) &mdash;{" "}
            precision {Number(trainedSvmData.precision).toFixed(2)},
            recall {Number(trainedSvmData.recall).toFixed(2)},
            F1 {Number(trainedSvmData.f1).toFixed(2)}{" "}
            <ReactTimeAgo date={trainedSvmData.date} timeStyle="mini"/> ago
          </div>}
        </div>
      </PopoverBody>
    );
  };
  return <>
    {orderingMode === "svm" &&
     <Popover
       placement="bottom"
       isOpen={!clusterIsOpen &&
               (svmPopoverOpen || svmIsTraining || !!!(trainedSvmData))}
       target="ordering-mode"
       trigger="hover"
       toggle={() => p.setSvmPopoverOpen(!svmPopoverOpen)}
       fade={false}
       popperClassName={`svm-popover ${svmIsTraining ? "loading" : ""}`}
     >
       {svmPopoverBody}
     </Popover>}
    {orderingMode === "knn" && <KnnPopover
                                 images={p.knnImages}
                                 dispatch={p.knnImagesDispatch}
                                 generateEmbedding={p.generateEmbedding}
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
  </>;
};

function ImageClusterViewer(props) {
  let datasetInfo = props.datasetInfo;
  let isLoading = props.isLoading;
  let queryResultData = props.queryResultData;
  let clusters = props.clusters;
  let clusteringStrength = props.clusteringStrength;
  let setSelection = props.setSelection;
  let setClusterIsOpen = props.setClusterIsOpen;
  let setPage = props.setPage;
  let page = props.page;
  let queryResultSet = props.queryResultSet;
  return (<Container fluid>
    {(!!!(datasetInfo.isNotLoaded) && !isLoading && queryResultSet.num_results == 0) &&
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
    {(!!!(datasetInfo.isNotLoaded) && queryResultSet.num_results > PAGE_SIZE) &&
     <div className="mt-4 d-flex justify-content-center">
       <ReactPaginate
         pageCount={Math.ceil(queryResultSet.num_results / PAGE_SIZE)}
         containerClassName="pagination"
         previousClassName="page-item"
         previousLinkClassName="page-link"
         nextClassName="page-item"
         nextLinkClassName="page-link"
         activeClassName="active"
         pageLinkClassName="page-link"
         pageClassName="page-item"
         breakClassName="page-item"
         breakLinkClassName="page-link"
         forcePage={page}
         onPageChange={({ selected }) => setPage(selected)}
         disabledClassName="disabled"
       />
     </div>
    }
  </Container>);
};


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
    };
  }, [onDragEnter, onDragExit, onDrop]);

  //
  // USER AUTHENTICATION
  //

  const [username, setUsername_] = useState(
    window.localStorage.getItem("foragerUsername") || ""
  );
  const setUsername = (u) => {
    window.localStorage.setItem("foragerUsername", u);
    setUsername_(u);
  }

  //
  // CLUSTER FOCUS MODAL
  //

  const [selection, setSelection] = useState({});
  const [clusterIsOpen, setClusterIsOpen] = useState(false);

  // Load dataset info on initial page load
  const [datasetName, setDatasetName] = useState("waymo");
  const [datasetInfo, setDatasetInfo] = useState({
    isNotLoaded: true,
    categories: [],
    index_id: null,
    num_train: 0,
    num_val: 0,
  });

  const getDatasetInfo = async () => {
    const url = new URL(`${endpoints.getDatasetInfo}/${datasetName}`);
    let _datasetInfo = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    setDatasetInfo(_datasetInfo);
    setIsLoading(true);
  }

  useEffect(getDatasetInfo, [datasetName]);

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
  const [split, setSplit] = useState(splits[0].id);
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
  const [svmModel, setSvmModel] = useState(null);
  const [svmIsTraining, setSvmIsTraining] = useState(false);
  const [trainedSvmData, setTrainedSvmData] = useState(null);

  const [rankingModel, setRankingModel] = useState(null);

  const [captionQuery, setCaptionQuery] = useState("");
  const [captionQueryEmbedding, setCaptionQueryEmbedding] = useState("");

  const [subset, setSubset_] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [queryResultSet, setQueryResultSet] = useState({
    id: null,
    num_results: 0,
    type: null,
  });
  const [queryResultData, setQueryResultData] = useState({
    images: [],
    clustering: [],
  });

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
    if (svmIsTraining) trainSvm().finally(() => setSvmIsTraining(false));
  }, [svmIsTraining]);

  const runQuery = async () => {
    setClusterIsOpen(false);
    setSelection({});

    let url;
    let body = {
      split: split,
      index_id: datasetInfo.index_id,
      include: datasetIncludeTags.map(t => `${t.category}:${t.value}`),
      exclude: datasetExcludeTags.map(t => `${t.category}:${t.value}`),
      subset: subset.map(im => im.id),
      score_min: scoreRange[0] / 100,
      score_max: scoreRange[1] / 100,
    };

    if (orderingMode === "id" || orderingMode === "random") {
      url = new URL(`${endpoints.queryImages}/${datasetName}`);
      body.order = orderingMode;
    } else if (orderingMode === "knn") {
      url = new URL(`${endpoints.queryKnn}/${datasetName}`);
      body.embeddings = Object.values(knnImages).map(i => i.embedding);
      body.use_full_image = !knnUseSpatial;
    } else if (orderingMode === "svm") {
      url = new URL(`${endpoints.querySvm}/${datasetName}`);
      body.svm_vector = trainedSvmData.svm_vector;
      if (svmModel) body.model = svmModel.with_output.model_id;
    } else if (orderingMode === "dnn") {
      url = new URL(`${endpoints.queryRanking}/${datasetName}`);
      body.model = rankingModel.with_output.model_id;
    } else if (orderingMode === "clip") {
      url = new URL(`${endpoints.queryKnn}/${datasetName}`);
      body.embeddings = [captionQueryEmbedding];
      body.model = "clip";
      body.use_dot_product = true;
    } else {
      console.error(`Query type (${orderingMode}) not implemented`);
      return;
    }
    const resultSet = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(r => r.json());

    if ((resultSet.type === "knn" || resultSet.type === "svm") &&
        (queryResultSet.type !== "knn" && queryResultSet.type !== "svm")) {
      setOrderByClusterSize(false);
    }

    setPage(0);
    setQueryResultSet(resultSet);
  };

  const [page, setPage] = useState(0);
  const [pageIsLoading, setPageIsLoading] = useState(false);

  const getPage = async () => {
    if (queryResultSet.num_results === 0) {
      setQueryResultData({
        images: [],
        clustering: [],
      });
      return;
    };

    let url = new URL(`${endpoints.getResults}/${datasetName}`);
    url.search = new URLSearchParams({
      index_id: datasetInfo.index_id,
      result_set_id: queryResultSet.id,
      offset: page * PAGE_SIZE,
      num: PAGE_SIZE,
    }).toString();
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

    window.scrollTo(0, 0);

    setQueryResultData({
      images,
      clustering: results.clustering,
    });
  };

  useEffect(() => {
    if (isLoading) runQuery().finally(() => setIsLoading(false));
  }, [isLoading]);

  useEffect(() => {
    setPageIsLoading(true);
    getPage().finally(() => setPageIsLoading(false));
  }, [page, queryResultSet]);

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
  const setMode = (mode) => {
    setMode_(mode);
    if (svmPopoverRepositionFunc.current) svmPopoverRepositionFunc.current();
  };

  const [labelModeCategory, setLabelModeCategory] = useState(null);  // label mode
  const [modelInfo, setModelInfo] = useState([]);  // train mode

   //
   // BULK TAG MODAL
   //
   const [bulkTagModalIsOpen, setBulkTagModalIsOpen] = useState(false);
   const toggleBulkTag = () => setBulkTagModalIsOpen(!bulkTagModalIsOpen);

  //
  // RENDERING
  //
  let mainHeaderProps = {
    username: username,
    setUsername: setUsername,
    datasetName: datasetName,
    datasetInfo: datasetInfo,
    setCategories: setCategories,
    clusterIsOpen: clusterIsOpen,
    setClusterIsOpen: setClusterIsOpen,
    clusteringStrength: clusteringStrength,
    selection: selection,
    setSelection: setSelection,
    clusters: clusters,
    knnImagesDispatch: knnImagesDispatch,
    setOrderingMode: setOrderingMode,
    generateEmbedding: generateEmbedding,
    setSubset: setSubset,
    mode: mode,
    setMode: setMode,
    labelModeCategory: labelModeCategory,
    queryResultSet: queryResultSet,
    bulkTagModalIsOpen: bulkTagModalIsOpen,
    toggleBulkTag: toggleBulkTag,
  };
  let queryBarProps = {
    datasetInfo: datasetInfo,
    setCategories: setCategories,
    split: split,
    setSplit: setSplit,
    datasetIncludeTags: datasetIncludeTags,
    setDatasetIncludeTags: setDatasetIncludeTags,
    datasetExcludeTags: datasetExcludeTags,
    setDatasetExcludeTags: setDatasetExcludeTags,
    setIsLoading: setIsLoading,
    orderingMode: orderingMode,
    setOrderingMode: setOrderingMode,
    orderingModes: orderingModes,
    trainedSvmData: trainedSvmData,
    knnImages: knnImages,
    rankingModel: rankingModel,
    captionQueryEmbedding: captionQueryEmbedding,
    queryResultSet: queryResultSet,
    scoreRange: scoreRange,
    setScoreRange: setScoreRange,
    subset: subset,
    setSubset: setSubset,
    orderByClusterSize: orderByClusterSize,
    setOrderByClusterSize: setOrderByClusterSize,
    clusteringStrength: clusteringStrength,
    setClusteringStrength: setClusteringStrength,
    recluster: recluster,
    bulkTagModalIsOpen: bulkTagModalIsOpen,
    toggleBulkTag: toggleBulkTag,
    username: username,
  };
  let orderingModeProps = {
    datasetInfo: datasetInfo,
    setCategories: setCategories,
    svmPopoverRepositionFunc: svmPopoverRepositionFunc,
    svmPosTags: svmPosTags,
    setSvmPosTags: setSvmPosTags,
    svmNegTags: svmNegTags,
    setSvmNegTags: setSvmNegTags,
    trainedSvmData: trainedSvmData,
    setTrainedSvmData: setTrainedSvmData,
    svmIsTraining: svmIsTraining,
    setSvmIsTraining: setSvmIsTraining,
    svmModel: svmModel,
    setSvmModel: setSvmModel,
    modelInfo: modelInfo,
    svmAugmentNegs: svmAugmentNegs,
    setSvmAugmentNegs: setSvmAugmentNegs,
    svmAugmentIncludeTags: svmAugmentIncludeTags,
    setSvmAugmentIncludeTags: setSvmAugmentIncludeTags,
    svmAugmentExcludeTags: svmAugmentExcludeTags,
    setSvmAugmentExcludeTags: setSvmAugmentExcludeTags,
    orderingMode: orderingMode,
    clusterIsOpen: clusterIsOpen,
    svmPopoverOpen: svmPopoverOpen,
    setSvmPopoverOpen: setSvmPopoverOpen,

    knnImages: knnImages,
    knnImagesDispatch: knnImagesDispatch,
    generateEmbedding: generateEmbedding,
    knnUseSpatial: knnUseSpatial,
    setKnnUseSpatial: setKnnUseSpatial,
    hasDrag: hasDrag,

    rankingModel: rankingModel,
    setRankingModel: setRankingModel,

    captionQuery: captionQuery,
    setCaptionQuery: setCaptionQuery,
    captionQueryEmbedding: captionQueryEmbedding,
    setCaptionQueryEmbedding: setCaptionQueryEmbedding,
  };

  return (
    <div className={`main ${(isLoading || pageIsLoading) ? "loading" : ""}`}>
      <MainHeader {...mainHeaderProps}/>
      <div className="border-bottom py-2 mode-container"
        style={{display: mode !== "explore" ? "block" : "none"}}>
        <Container fluid>
          <LabelPanel
            categories={datasetInfo.categories}
            setCategories={setCategories}
            category={labelModeCategory}
            setCategory={setLabelModeCategory}
            isVisible={mode === "label"}
          />
          <TrainPanel
            datasetName={datasetName}
            datasetInfo={datasetInfo}
            modelInfo={modelInfo}
            setModelInfo={setModelInfo}
            isVisible={mode === "train"}
            username={username}
            disabled={!!!(username)}
            categories={datasetInfo.categories}
          />
          <ValidatePanel
            datasetName={datasetName}
            modelInfo={modelInfo}
            datasetInfo={datasetInfo}
            username={username}
            isVisible={mode === "validate"}
          />
        </Container>
      </div>
      <div className="app">
        <QueryBar {...queryBarProps}/>
        <OrderingModeSelector {...orderingModeProps}/>
        <ImageClusterViewer
          datasetInfo={datasetInfo}
          isLoading={isLoading || pageIsLoading}
          queryResultData={queryResultData}
          queryResultSet={queryResultSet}
          clusters={clusters}
          clusteringStrength={clusteringStrength}
          setSelection={setSelection}
          setClusterIsOpen={setClusterIsOpen}
          setPage={setPage}
          page={page} />
      </div>
    </div>
  );
}

export default App;
