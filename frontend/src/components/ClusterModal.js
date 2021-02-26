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
import { faMousePointer } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";
import difference from "lodash/difference";
import intersection from "lodash/intersection";
import uniq from "lodash/uniq";
import union from "lodash/union";

import ImageGrid from "./ImageGrid";

const endpoints = fromPairs(toPairs({
  getAnnotations: 'get_annotations_v2',
  addAnnotations: 'add_annotations_v2',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const imageGridSizes = [
  {label: "small", size: 125},
  {label: "medium", size: 250},
  {label: "large", size: 375},
];

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
  setSubset,
}) => {
  const typeaheadRef = useRef();

  const selectedCluster = (selection.cluster !== undefined &&
                           selection.cluster < clusters.length) ?
                          clusters[selection.cluster] : undefined;
  const isSingletonCluster = (selectedCluster !== undefined &&
                              selectedCluster.length === 1);

  let selectedImage;
  if (isSingletonCluster) {
    selectedImage = (selectedCluster !== undefined) ? selectedCluster[0] : undefined;
  } else {
    selectedImage = (selectedCluster !== undefined &&
                     selection.image !== undefined &&
                     selection.image < selectedCluster.length) ?
                    selectedCluster[selection.image] : undefined;
  }
  const isClusterView = (selectedCluster !== undefined &&
                         selectedImage === undefined);
  const isImageView = !isSingletonCluster && !isClusterView;

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

  //
  // IMAGE SELECTION
  //

  const [excludedImageIndexes, setExcludedImageIndexes] = useState({});
  const [imageGridSize, setImageGridSize_] = useState(imageGridSizes[0]);

  const excludeNone = (e) => {
    setExcludedImageIndexes({});
    if (e) e.preventDefault();
  };

  const excludeAll = (e) => {
    setExcludedImageIndexes(fromPairs(selectedCluster.map((_, i) => [i, true])));
    if (e) e.preventDefault();
  };

  useEffect(() => {
    excludeNone();
  }, [selectedCluster]);

  const handleGalleryClick = (e, i) => {
    if (e.shiftKey) {
      toggleImageSelection(i);
    } else {
      setSelection({
        cluster: selection.cluster,
        image: i
      });
    }
  };

  const toggleImageSelection = (i, e) => {
    let newExcludedImageIndexes = {...excludedImageIndexes};
    newExcludedImageIndexes[i] = !!!(newExcludedImageIndexes[i]);
    setExcludedImageIndexes(newExcludedImageIndexes);
    if (e) e.preventDefault();
  }

  const setImageGridSize = (size, e) => {
    setImageGridSize_(size);
    e.preventDefault();
  };

  //
  // TAGGING
  //

  const [isLoading, setIsLoading] = useState(false);

  const getImageTags = im => (annotations[im.id] || []);
  let selectedTags = [];
  if (selectedImage !== undefined) {
    selectedTags = getImageTags(selectedImage);
  } else if (selectedCluster !== undefined) {
    selectedTags = intersection(...(selectedCluster.flatMap((im, i) =>
      excludedImageIndexes[i] ? [] : [getImageTags(im)])));
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
      selectedCluster.flatMap((im, i) => excludedImageIndexes[i] ? [] : [im.id]);

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

  //
  // KEY BINDINGS
  //

  const handleKeyDown = useCallback((e) => {
    if (!isOpen) return;
    const { key } = e;
    let caught = true;
    if (isClusterView && key === "ArrowDown") {
      // Switch to image view
      setSelection({
        cluster: selection.cluster,
        image: 0
      });
    } else if (isImageView && key === "ArrowUp") {
      // Switch to cluster view
      setSelection({
        cluster: selection.cluster,
      });
    } else if (isImageView && key === "ArrowLeft") {
      // Previous image
      setSelection({
        cluster: selection.cluster,
        image: Math.max(selection.image - 1, 0)
      });
    } else if (isImageView && key === "ArrowRight") {
      // Next image
      setSelection({
        cluster: selection.cluster,
        image: Math.min(selection.image + 1, clusters[selection.cluster].length - 1)
      });
    } else if (key === "ArrowLeft") {
      // Previous cluster
      setSelection({
        cluster: Math.max(selection.cluster - 1, 0),
        image: selection.image && 0
      });
    } else if (key === "ArrowRight") {
      // Next cluster
      setSelection({
        cluster: Math.min(selection.cluster + 1, clusters.length - 1),
      });
    } else if (isImageView && key === "s") {
      // Toggle selection
      toggleImageSelection(selection.image);
    } else if (key !== "ArrowDown" && key !== "ArrowUp") {
      caught = false;
    }
    if (caught) {
      e.preventDefault();
      typeaheadRef.current.blur();
      typeaheadRef.current.hideMenu();
    }
  }, [isOpen, isClusterView, isImageView, clusters, selection, setSelection, typeaheadRef, excludedImageIndexes]);

  const handleTypeaheadKeyDown = (e) => {
    const { key } = e;
    if (key === "s") e.stopPropagation();
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
    if (isClusterView) {
      header += ` (${selectedCluster.length} images)`;
    } else if (isSingletonCluster && !isImageOnly) {
      header += " (1 image)";
    } else if (isImageView) {
      header += `, image ${selection.image + 1} of ${clusters[selection.cluster].length}`;
    }
  }

  return (
    <Modal
      isOpen={isOpen}
      toggle={() => setIsOpen(false)}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="full"
      className={`cluster-modal ${isLoading ? "loading" : ""}`}
    >
      {(selectedCluster !== undefined) && <>
        <ModalHeader toggle={() => setIsOpen(false)}>
          <span>{header}</span>
        </ModalHeader>

        <ModalBody>
          <p>
            <kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move between {(isImageView || isImageOnly) ? "images" : "clusters"}
            {isClusterView && <>,{" "}
              <kbd>&darr;</kbd> or <FontAwesomeIcon icon={faMousePointer} /> to go into image view, <kbd>shift</kbd> <FontAwesomeIcon icon={faMousePointer} /> to toggle image selection</>}
            {isImageView && <>,{" "}
              <kbd>&uarr;</kbd> to go back to cluster view, <kbd>s</kbd> or <FontAwesomeIcon icon={faMousePointer} /> to toggle image selection</>}
          </p>
          <Form>
            <FormGroup className="d-flex flex-row align-items-center mb-2">
              <Typeahead
                multiple
                allowNew
                id="image-tag-bar"
                className="typeahead-bar"
                placeholder="Image tags"
                disabled={isReadOnly || (isClusterView && selectedCluster.length === Object.values(excludedImageIndexes).filter(Boolean).length)}
                options={tags}
                selected={selectedTags}
                onChange={onTagsChanged}
                ref={typeaheadRef}
                onBlur={() => typeaheadRef.current.hideMenu()}
                onKeyDown={handleTypeaheadKeyDown}
              />
              {(isClusterView) ?
                <Button color="light" className="ml-2" onClick={() => setSubset(selectedCluster)}>
                  Descend into cluster
                </Button> :
                <Button color="warning" className="ml-2" onClick={() => findSimilar(selectedImage)}>
                  Find similar images
                </Button>}
            </FormGroup>
          </Form>
          {selectedImage !== undefined ?
            <ProgressiveImage
              src={selectedImage.src}
              placeholder={selectedImage.thumb}
            >
              {src => {
                if (isImageView) {
                  const selected = !!!(excludedImageIndexes[selection.image]);
                  return (
                    <a href="#" onClick={(e) => toggleImageSelection(selection.image, e)} className="selectable-image">
                      <img className="w-100" src={src} />
                      <div className={`state rbt-token alert-${selected ? "success": "secondary"}`}>
                        {selected ? "S" : "Not s"}elected
                      </div>
                    </a>);
                } else {
                  return <img className="main w-100" src={src} />;
                }
              }}
            </ProgressiveImage> :
            <>
              <div className="mb-1 text-small text-secondary font-weight-normal">
                Selected {selectedCluster.length - Object.values(excludedImageIndexes).filter(Boolean).length}{" "}
                of {selectedCluster.length} images (
                <a href="#" className="text-secondary" onClick={excludeNone}>select all</a>,{" "}
                <a href="#" className="text-secondary" onClick={excludeAll}>deselect all</a>){" "}
                (thumbnails: {imageGridSizes.map((size, i) =>
                  <>
                    <a href="#" className="text-secondary" onClick={(e) => setImageGridSize(size, e)}>{size.label}</a>
                    {(i < imageGridSizes.length - 1) ? ", " : ""}
                  </>
                )})
              </div>
              <ImageGrid
                images={selectedCluster}
                onClick={handleGalleryClick}
                selectedPred={i => !!!(excludedImageIndexes[i])}
                minRowHeight={imageGridSize.size}
                imageAspectRatio={3/2}
              />
            </>
          }
        </ModalBody>
      </>}
    </Modal>
  );
}

export default ClusterModal;
