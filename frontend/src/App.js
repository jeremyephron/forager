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

const getWaymoImageSpec = (id) => {
  return {
    name: `${id}_front.jpeg`,
    url: `https://storage.googleapis.com/foragerml/waymo/train/${id}_front.jpeg`,
  };
}

const ImageStack = ({ id, onClick, images }) => {
  return (
    <a className="stack" onClick={onClick}>
      {times(Math.min(4, images.length), (i) =>
        <img key={`stack-${i}`} className="thumb" src={images[i].url}></img>
      )}
      <div className="label">
        <b>Cluster {id + 1}</b> ({images.length} image{images.length !== 1 && "s"})
      </div>
    </a>
  );
}

const App = () => {
  const [source, setSource] = useState(sources[0].id);
  const [datasetQuery, setDatasetQuery] = useState([tags[0]]);
  const [googleQuery, setGoogleQuery] = useState("");
  const [clusteringStrength, setClusteringStrength] = useState(0);
  const [selection, setSelection] = useState({});
  const [isOpen, setIsOpen] = useState(false);
  const [clusters, setClusters] = useState(Array(100).fill(imageIds.slice(0, 5).map(getWaymoImageSpec)));

  const handleKeyDown = useCallback((e) => {
    if (isOpen) {
      const { key } = e;
      let caught = true;
      if (key === 'ArrowDown') {
        setSelection({cluster: Math.min(selection.cluster + 1, clusters.length - 1), image: 0});
      } else if (key === 'ArrowUp') {
        setSelection({cluster: Math.max(selection.cluster - 1, 0), image: 0});
      } else if (key === 'ArrowRight') {
        setSelection({...selection, image: Math.min(selection.image + 1, clusters[selection.cluster].length - 1)});
      } else if (key === 'ArrowLeft') {
        setSelection({...selection, image: Math.max(selection.image - 1, 0)});
      } else {
        caught = false;
      }
      if (caught) e.preventDefault();
    }
  }, [isOpen, setSelection, selection]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [handleKeyDown]);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <div>
      <Modal isOpen={isOpen} toggle={toggle} size="lg">
        {selection.cluster !== undefined && selection.image !== undefined &&
          <>
            <ModalHeader toggle={toggle}>
              Cluster {selection.cluster + 1} of {clusters.length},
              image {selection.image + 1} of {clusters[selection.cluster].length}
              <span className="text-muted"> ({clusters[selection.cluster][selection.image].name})</span>
            </ModalHeader>
            <ModalBody>
              <p>
                <b>Key bindings: </b> use <kbd>&uarr;</kbd> <kbd>&darr;</kbd> to move
                between clusters, <kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move between images
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
                    options={tags.filter((t) => !t.hide)}
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
                <select className="custom-select mr-2" onChange={(e) => setSource(e.target.value)}>
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
                    onChange={(selected) => setDatasetQuery(selected)}
                  />)
                } else if (source === "google") {
                  return (
                    <Input
                      type="text"
                      placeholder="Query"
                      value={googleQuery}
                      onChange={(e) => setGoogleQuery(e.target.value)}
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
                  value={clusteringStrength} onChange={(e) => setClusteringStrength(e.target.value)} />
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
