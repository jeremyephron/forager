import React, { useState, useRef } from "react";
import {
  Button,
  Form,
  FormGroup,
  Modal,
  ModalHeader,
  ModalBody,
} from "reactstrap";
import { Typeahead } from "react-bootstrap-typeahead";
import ProgressiveImage from "react-progressive-image";
import uniq from "lodash/uniq";

const ClusterModal = ({
  isOpen,
  setIsOpen,
  isImageOnly,
  selection,
  clusters,
  findSimilar,
  isReadOnly,
  tags,
  annotations,
  onTagsChanged,
}) => {
  const typeaheadRef = useRef();

  const selectedImage = (
    selection.cluster !== undefined && selection.image !== undefined &&
    selection.cluster < clusters.length &&
    selection.image < clusters[selection.cluster].length
  ) ? clusters[selection.cluster][selection.image] : undefined;
  const selectedTags = (selectedImage ? annotations[selectedImage.id] : undefined) || [];

  const parseNewTags = (newTags) => uniq(newTags.map(t => (t.label || t)));

  const handleKeyDown = (e) => {
    const { key } = e;
    if (key === "ArrowDown" ||
        key === "ArrowUp" ||
        key === "ArrowLeft" ||
        key === "ArrowRight") {
      typeaheadRef.current.blur();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      toggle={() => setIsOpen(false)}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="lg"
    >
      {!!(selectedImage) && <>
        <ModalHeader toggle={() => setIsOpen(false)}>
          {isImageOnly ?
            <span>
              Image {selection.cluster + 1} of {clusters.length}
            </span> :
            <span>
              Cluster {selection.cluster + 1} of {clusters.length},
              image {selection.image + 1} of {clusters[selection.cluster].length}
            </span>}
          <span className="text-muted"> ({selectedImage.name})</span>
        </ModalHeader>
        <ModalBody>
          <p>
            <b>Key bindings: </b> use <kbd>&uarr;</kbd> <kbd>&darr;</kbd>
            {isImageOnly ? " or " : " to move between clusters, "}
            <kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move between images
          </p>
          <ProgressiveImage
            src={selectedImage.url}
            placeholder={selectedImage.thumb}
          >
            {src => <img className="w-100" src={src} />}
          </ProgressiveImage>
          <img  />
          <Form>
            <FormGroup className="mt-2 mb-0 d-flex flex-row align-items-center">
              <Button color="warning" className="mr-2" onClick={findSimilar}>Find similar images</Button>
              <Typeahead
                multiple
                allowNew
                id="image-tag-bar"
                className="typeahead-bar"
                placeholder="Image tags"
                disabled={isReadOnly}
                options={tags}
                selected={selectedTags.map(t => t.category)}
                onChange={(newTags) => onTagsChanged(selectedImage, selectedTags, parseNewTags(newTags))}
                onKeyDown={handleKeyDown}
                ref={typeaheadRef}
              />
            </FormGroup>
          </Form>
        </ModalBody>
      </>}
    </Modal>
  );
}

export default ClusterModal;
