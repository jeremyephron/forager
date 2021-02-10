import React, { useState, useEffect, useCallback } from "react";
import {
  Container,
  Row,
  Col,
  Button,
  Navbar,
  NavbarBrand,
  Form,
  FormGroup,
  Input,
  Modal,
  ModalHeader,
  ModalBody,
} from "reactstrap";
import { Typeahead } from "react-bootstrap-typeahead";
import { ReactSVG } from "react-svg";
import times from "lodash/times";

import imageIds from "./data/waymo";

import "react-bootstrap-typeahead/css/Typeahead.css";
import "./scss/theme.scss";

var performClustering = require("hierarchical-clustering");
var computeSimilarity = require("compute-cosine-similarity");

const sources = [
  {id: "dataset", label: "Dataset"},
  {id: "google", label: "Google"},
]

const tags = [
  {id: "full_dataset", label: "full dataset", hide: true},
  {id: "50_dataset", label: "50% dataset", hide: true},
  {id: "10_dataset", label: "10% dataset", hide: true},
  {id: "1_dataset", label: "1% dataset", hide: true},
  {id: "pickup_truck", label: "pickup truck"},
  {id: "house_flag", label: "house flag"},
  {id: "pride_flag", label: "pride flag"},
  {id: "pride_flag", label: "street flag"},
  {id: "garbage_bin", label: "garbage bin"},
]

const images = imageIds.slice(0, 1000).map(id => {
  return {
    name: `${id}_front.jpeg`,
    url: `https://storage.googleapis.com/foragerml/waymo/train/${id}_front.jpeg`,
    embedding: [...new Array(256)].map(() => Math.random() * 10 - 5)  // random for now
  };
})

const imageSimilarity = (a, b) => computeSimilarity(a.embedding, b.embedding);

const ImageStack = ({ id, onClick, images, showLabel }) => {
  return (
    <a className="stack" onClick={onClick}>
      {times(Math.min(4, images.length), (i) =>
        <img key={`stack-${i}`} className="thumb" src={images[i].url}></img>
      )}
      {showLabel && <div className="label">
        <b>Cluster {id + 1}</b> ({images.length} image{images.length !== 1 && "s"})
      </div>}
    </a>
  );
}

// TEMPORARY until we have a real clustering algorithm in
let initialClusters = [];
for (let i = 0; i < 100; i++) {
  let cluster = [];
  for (let j = 0; j < 10; j++) {
    cluster.push(images[i * 10 + j]);
  }
  initialClusters.push(cluster);
}

const App = () => {
  const [source, setSource] = useState(sources[0].id);
  const [datasetQuery, setDatasetQuery] = useState([tags[0]]);
  const [googleQuery, setGoogleQuery] = useState("");
  const [clusteringStrength, setClusteringStrength] = useState(50);
  const [selection, setSelection] = useState({});
  const [isOpen, setIsOpen] = useState(false);
  const [clusters, setClusters] = useState([]);

  const handleKeyDown = useCallback(e => {
    if (isOpen) {
      const { key } = e;
      let caught = true;
      if (key === "ArrowDown" || (clusteringStrength == 0 && key === "ArrowRight")) {
        setSelection({cluster: Math.min(selection.cluster + 1, clusters.length - 1), image: 0});
      } else if (key === "ArrowUp" || (clusteringStrength == 0 && key === "ArrowLeft")) {
        setSelection({cluster: Math.max(selection.cluster - 1, 0), image: 0});
      } else if (key === "ArrowRight") {
        setSelection({...selection, image: Math.min(selection.image + 1, clusters[selection.cluster].length - 1)});
      } else if (key === "ArrowLeft") {
        setSelection({...selection, image: Math.max(selection.image - 1, 0)});
      } else {
        caught = false;
      }
      if (caught) e.preventDefault();
    }
  }, [isOpen, clusteringStrength, setSelection, selection]);

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [handleKeyDown]);

  const recluster = () => {
    if (clusteringStrength == 0) {
      setClusters(images.map(i => [i]));
    } else {
      // TODO(mihirg): Figure out a way to do this that doesn't crash the browser
      // TODO(mihirg): Cache distance matrix for faster re-clustering
      setClusters(initialClusters);
      // const levels = performClustering({
      //   input: images,
      //   distance: imageSimilarity,
      //   linkage: "average",
      //   maxLinkage: clusteringStrength / 100,
      // });
      // const clusterIndices = levels[levels.length - 1].clusters;
      // setClusters(clusterIndices.map(cluster => cluster.map(index => images[index])));
    }
  };
  useEffect(recluster, []);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <div>
      <Modal isOpen={isOpen} toggle={toggle} size="lg">
        {selection.cluster !== undefined && selection.image !== undefined &&
          <>
            <ModalHeader toggle={toggle}>
              {(clusteringStrength > 0) ?
                <span>
                  Cluster {selection.cluster + 1} of {clusters.length},
                  image {selection.image + 1} of {clusters[selection.cluster].length}
                </span>:
                <span>
                  Image {selection.cluster + 1} of {clusters.length}
                </span>}
              <span className="text-muted"> ({clusters[selection.cluster][selection.image].name})</span>
            </ModalHeader>
            <ModalBody>
              <p>
                <b>Key bindings: </b> use <kbd>&uarr;</kbd> <kbd>&darr;</kbd>
                {(clusteringStrength > 0) ? " to move between clusters, " : " or "}
                <kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move between images
              </p>
              <img src={clusters[selection.cluster][selection.image].url} />
              <Form>
                <FormGroup className="mt-2 mb-0 d-flex flex-row align-items-center">
                  <Button color="warning" className="mr-2" onClick={() => {}}>Find similar images</Button>
                  <Typeahead
                    multiple
                    allowNew
                    id="image-tag-bar"
                    className="typeahead-bar"
                    placeholder="Image tags"
                    options={tags.filter(t => !t.hide)}
                  />
                </FormGroup>
              </Form>
            </ModalBody>
          </>}
      </Modal>

      <Navbar color="primary" className="text-light" dark>
        <Container fluid>
          <span>
            <NavbarBrand href="/">Forager</NavbarBrand>
            <NavbarBrand className="font-weight-normal">waymo_train</NavbarBrand>
          </span>
          <span>
            mihirg@stanford.edu (<a href="#">Logout</a>)
          </span>
        </Container>
      </Navbar>
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
                  return (<Typeahead
                    multiple
                    id="dataset-query-bar"
                    className="typeahead-bar"
                    placeholder="Query tags"
                    options={tags}
                    selected={datasetQuery}
                    onChange={selected => setDatasetQuery(selected)}
                  />)
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
                <select className="custom-select mx-2">
                  <option value="default" selected>Default order</option>
                  <option value="random">Random order</option>
                  <option value="knn" disabled>KNN</option>
                </select>
                <ReactSVG className="icon" src="assets/arrow-caret.svg" />
              </FormGroup>
              <Button color="primary" className="mr-4">Run query</Button>
              <div>
                <span className="text-nowrap">Clustering strength:</span>
                <input className="custom-range" type="range" min="0" max="100"
                  value={clusteringStrength}
                  onChange={e => setClusteringStrength(e.target.value)}
                  onMouseUp={recluster} />
              </div>
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
                    setSelection({cluster: i, image: 0});
                    setIsOpen(true);
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
