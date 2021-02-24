import React, { useState, useCallback, useEffect, useRef } from "react";
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
import Gallery from "react-photo-gallery";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";
import difference from "lodash/difference";
import intersection from "lodash/intersection";
import uniq from "lodash/uniq";
import union from "lodash/union";

const endpoints = fromPairs(toPairs({
  getAnnotations: 'get_annotations_v2',
  addAnnotations: 'add_annotations_v2',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const ClusterModal = ({
  isOpen,
  setIsOpen,
  isImageOnly,
  isReadOnly,
  selection,
  setSelection,
  clusters,
  findSimilar,
  tags,
  setTags,
  username,
}) => {
  const typeaheadRef = useRef();

  const selectedCluster = (selection.cluster !== undefined &&
                           selection.cluster < clusters.length) ?
                          clusters[selection.cluster] : undefined;
  let selectedImage;
  if (isImageOnly) {
    selectedImage = (selectedCluster !== undefined) ? selectedCluster[0] : undefined;
  } else {
    selectedImage = (selectedCluster !== undefined &&
                     selection.image !== undefined &&
                     selection.image < selectedCluster.length) ?
                    selectedCluster[selection.image] : undefined;
  }
  const isClusterView = (!isImageOnly &&
                         selectedCluster !== undefined &&
                         selectedImage === undefined);

  //
  // DATA CONNECTIONS
  //

  const [annotations, setAnnotations] = useState({});

  // Reload annotations whenever there's a new result set
  useEffect(async () => {
    if (clusters.length === 0) return;
    let annotationsUrl = new URL(endpoints.getAnnotations);
    annotationsUrl.search = new URLSearchParams({
      identifiers: clusters.map(cl => cl.map(im => im.id)).flat()
    }).toString();
    setAnnotations(await fetch(annotationsUrl, {
      method: "GET",
    }).then(r => r.json()));
  }, [clusters]);

  useEffect(() => console.log(annotations), [annotations]);

  //
  // TAGGING
  //

  const [isLoading, setIsLoading] = useState(false);

  const getImageTags = im => (annotations[im.id] || []);
  let selectedTags = [];
  if (selectedImage !== undefined) {
    selectedTags = getImageTags(selectedImage);
  } else if (selectedCluster !== undefined) {
    selectedTags = intersection(...(selectedCluster.map(getImageTags)));
  }

  // Add or remove tags whenever the typeahead value changes
  const addAnnotations = async (category, value, identifiers) => {
    const url = new URL(endpoints.addAnnotations);
    const body = {
      user: username,
      category,
      value,
      identifiers
    };
    return fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
  }
  const onTagsChanged = async (newTags) => {
    newTags = uniq(newTags.map(t => (t.label || t)));
    const added = difference(newTags, selectedTags);
    const deleted = difference(selectedTags, newTags);
    const imageIds = (selectedImage !== undefined) ? [selectedImage.id] :
                                                     selectedCluster.map(im => im.id);

    let newAnnotations = {...annotations};
    for (const id of imageIds) {
      newAnnotations[id] = union(difference(annotations[id], deleted), added);
    }
    setIsLoading(true);
    setAnnotations(newAnnotations);
    setTags(union(tags, added));

    let addPromises = added.map(async t => addAnnotations(t, "positive", imageIds));
    let deletePromises = deleted.map(async t => addAnnotations(t, "negative", imageIds));
    await Promise.all([...addPromises, ...deletePromises]);

    setIsLoading(false);
  }

  const clearImageSelection = (e) => {
    setSelection({cluster: selection.cluster});
    if (e) e.preventDefault();
  }

  //
  // KEY BINDINGS
  //

  const handleKeyDown = useCallback((e) => {
    if (!isOpen) return;
    const { key } = e;
    let caught = true;
    if (key === "ArrowDown" || ((isImageOnly || isClusterView) && key === "ArrowRight")) {
      setSelection({
        cluster: Math.min(selection.cluster + 1, clusters.length - 1),
        image: selection.image && 0
      });
    } else if (key === "ArrowUp" || ((isImageOnly || isClusterView) && key === "ArrowLeft")) {
      setSelection({
        cluster: Math.max(selection.cluster - 1, 0),
        image: selection.image && 0
      });
    } else if (key === "ArrowRight") {
      setSelection({
        cluster: selection.cluster,
        image: Math.min(selection.image + 1, clusters[selection.cluster].length - 1)
      });
    } else if (key === "ArrowLeft") {
      setSelection({
        cluster: selection.cluster,
        image: Math.max(selection.image - 1, 0)
      });
    } else if (key === "c" && !isImageOnly && !isClusterView) {
      clearImageSelection();
    } else {
      caught = false;
    }
    if (caught) {
      e.preventDefault();
      typeaheadRef.current.blur();
      typeaheadRef.current.hideMenu();
    }
  }, [isOpen, isImageOnly, isClusterView, clusters, selection, setSelection, typeaheadRef]);

  const handleTypeaheadKeyDown = (e) => {
    if (!isOpen) return;
    const { key } = e;
    if (key === "c") e.stopPropagation();
  }

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [handleKeyDown]);

  //
  // RENDERING
  //

  let header;
  if (selectedCluster !== undefined) {
    header = `${isImageOnly ? "Image" : "Cluster"} ${selection.cluster + 1} of ${clusters.length}`;
    if (!isImageOnly && !isClusterView) header += `, image ${selection.image + 1} of ${clusters[selection.cluster].length}`;
  }

  return (
    <Modal
      isOpen={isOpen}
      toggle={() => setIsOpen(false)}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="lg"
      className={isLoading ? "loading" : ""}
    >
      {(selectedCluster !== undefined) && <>
        <ModalHeader toggle={() => setIsOpen(false)}>
          <span>{header}</span>
        </ModalHeader>

        <ModalBody>
          <p>
            <b>Key bindings: </b> <kbd>&uarr;</kbd> <kbd>&darr;</kbd>
            {(isImageOnly || isClusterView) ? " or " : " between clusters, "}
            <kbd>&larr;</kbd> <kbd>&rarr;</kbd> between
            {isClusterView ? " clusters" : " images"}
            {!isImageOnly && !isClusterView &&
              <>, <kbd>c</kbd> to go <a href="#" onClick={clearImageSelection}>back to cluster view</a></>}
          </p>
          <Form>
            <FormGroup className="d-flex flex-row align-items-center">
              <Typeahead
                multiple
                allowNew
                id="image-tag-bar"
                className="typeahead-bar"
                placeholder="Image tags"
                disabled={isReadOnly}
                options={tags}
                selected={selectedTags}
                onChange={onTagsChanged}
                ref={typeaheadRef}
                onKeyDown={handleTypeaheadKeyDown}
                onBlur={() => typeaheadRef.current.hideMenu()}
              />
              {(selectedImage !== undefined) &&
                <Button color="warning" className="ml-2" onClick={findSimilar}>Find similar images</Button>}
            </FormGroup>
          </Form>
          {selectedImage !== undefined ?
            <ProgressiveImage
              src={selectedImage.fullResSrc}
              placeholder={selectedImage.src}
            >
              {src => <img className="w-100" src={src} />}
            </ProgressiveImage> :
            <Gallery
              photos={selectedCluster}
              targetRowHeight={140}
              margin={1}
              onClick={(_, {index}) => setSelection({...selection, image: index})}
            />  // TODO(mihirg): Make images lazy load
          }
        </ModalBody>
      </>}
    </Modal>
  );
}

export default ClusterModal;
