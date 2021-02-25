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

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import "react-bootstrap-typeahead/css/Typeahead.css";
import "./scss/theme.scss";

import ImageStack from "./components/ImageStack";
import ClusterModal from "./components/ClusterModal";

var disjointSet = require("disjoint-set");

const sources = [
  {id: "dataset", label: "Dataset"},
  // {id: "google", label: "Google"},
]

const orderingModes = [
  {id: "default", label: "Default"},
  // {id: "random", label: "Random"},
  {id: "knn", label: "KNN", disabled: true},
]

const endpoints = fromPairs(toPairs({
  getDatasetInfo: 'get_dataset_info_v2',
  getNextImages: 'get_next_images_v2',
  queryKnn: 'query_knn_v2',
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
  // DATA CONNECTIONS
  //

  // Load dataset info on initial page load
  const [datasetName, setDatasetName] = useState("waymo_train_central");
  const [datasetInfo, setDatasetInfo] = useState({
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
  const [orderByClusterSize, setOrderByClusterSize] = useState(false);
  const [clusteringStrength, setClusteringStrength] = useState(50);

  const [knnImage, setKnnImage] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [queryResultData, setQueryResultData] = useState({images: [], clustering: []});

  const runQuery = async () => {
    let url;
    let params = {
      num: 1000,
      index_id: datasetInfo.index_id,
      include: datasetIncludeTags,
      exclude: datasetExcludeTags,
    }

    if (source == "dataset" && orderingMode == "default") {
      url = new URL(`${endpoints.getNextImages}/${datasetName}`);
      url.search = new URLSearchParams(params).toString();
    } else if (source == "dataset" && orderingMode == "knn") {
      url = new URL(`${endpoints.queryKnn}/${datasetName}`);
      url.search = new URLSearchParams({...params, image_ids: [knnImage.id]}).toString();
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
        fullResSrc: path,
        id: results.identifiers[i],
        src: `https://storage.googleapis.com/foragerml/thumbnails/${datasetInfo.index_id}/${id}.jpg`,
        width: 300,
        height: 200,
      };
    });

    setClusterIsOpen(false);
    setSelection({});
    setQueryResultData({
      images,
      clustering: results.clustering,
    });
    setIsLoading(false);
  };
  useEffect(() => {
    if (isLoading) runQuery();
  }, [isLoading]);

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
      let ds = disjointSet();
      for (let image of queryResultData.images) {
        ds.add(image);
      }
      for (let [a, b, dist] of queryResultData.clustering) {
        if (dist > clusteringStrength / 100) break;
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
      <Modal
        isOpen={loginIsOpen}
        toggle={() => setLoginIsOpen(false)}
        modalTransition={{ timeout: 25 }}
        backdropTransition={{ timeout: 75 }}
      >
        <ModalBody>
          <div className="m-xl-4 m-3">
            <div className="text-center mb-4">
              <h4 className="h3 mb-1">Welcome back</h4>
              <span>Enter your account details below</span>
            </div>
            <Form>
              <FormGroup>
                <Input
                  type="email"
                  placeholder="Email Address"
                  value={loginUsername}
                  onChange={(e) => setLoginUsername(e.target.value)}
                />
              </FormGroup>
              <FormGroup>
                <Input
                  type="password"
                  placeholder="Password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                />
              </FormGroup>
              <FormGroup>
                <Button block color="primary" type="submit" onClick={login}>Sign in</Button>
              </FormGroup>
            </Form>
          </div>
        </ModalBody>
      </Modal>
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
      />
      <Navbar color="primary" className="text-light mb-2" dark>
        <Container fluid>
          <span>
            <NavbarBrand href="/">Forager</NavbarBrand>
            <NavbarBrand className="font-weight-normal" id="dataset-name">{datasetName}</NavbarBrand>
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
                <select className="custom-select mx-2" onChange={e => setOrderingMode(e.target.value)}>
                  {orderingModes.map((m) => <option value={m.id} selected={orderingMode === m.id} disabled={m.disabled}>{m.label}</option>)}
                </select>
                <ReactSVG className="icon" src="assets/arrow-caret.svg" />
              </FormGroup>
              <Button color="primary" onClick={() => setIsLoading(true)}>Run query</Button>
            </Form>
            <Form className="mt-2 d-flex flex-row align-items-center">
              <div className="custom-switch mr-5 ml-auto">
                <Input type="checkbox" className="custom-control-input"
                  id="order-by-cluster-size-switch"
                  value={orderByClusterSize}
                  onChange={(e) => setOrderByClusterSize(e.target.checked)}
                />
                <label className="custom-control-label text-nowrap" for="order-by-cluster-size-switch">
                  Order by cluster size
                </label>
              </div>
              <label for="cluster-strength-slider" className="mb-0 mr-2 text-nowrap">
                Clustering strength:
              </label>
              <input className="custom-range" type="range" min="0" max="100"
                id="cluster-strength-slider"
                value={clusteringStrength}
                onChange={e => setClusteringStrength(e.target.value)}
                onMouseUp={recluster}
              />
            </Form>
          </Container>
        </div>
        <Container fluid>
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
