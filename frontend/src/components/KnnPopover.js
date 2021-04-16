import React, { useState, useEffect, useReducer } from "react";
import {
  Popover,
  PopoverBody,
  Alert,
  Input,
  Spinner,
} from "reactstrap";
import Dropzone from "react-dropzone";
import Emoji from "react-emoji-render";
import { v4 as uuidv4 } from "uuid";
import { faTimesCircle } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import some from "lodash/some";
import size from "lodash/size";

const KnnPopover = ({ images, dispatch, generateEmbedding, useSpatial, setUseSpatial, hasDrag }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toBase64 = file => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length !== 1) return;
    const uuid = uuidv4();
    const file = acceptedFiles[0];
    dispatch({
      type: "ADD_IMAGE_FILE",
      file,
      uuid,
    });
    generateEmbedding({image_data: await toBase64(file)}, uuid);
  };

  const isLoading = some(Object.values(images).map(i => !(i.embedding)));

  return (
    <Popover
      placement="bottom"
      isOpen={isOpen || hasDrag}
      target="ordering-mode"
      trigger="hover"
      toggle={() => setIsOpen(!isOpen)}
      fade={false}
      popperClassName="knn-popover"
    >
      <PopoverBody>
        {Object.entries(images).map(([uuid, image]) =>
          <div className="mb-1">
            <img
              className="w-100"
              key={uuid}
              src={image.src}
            />
            <FontAwesomeIcon
              icon={faTimesCircle}
              style={{cursor: "pointer"}}
              onClick={() => dispatch({
                type: "DELETE_IMAGE",
                uuid,
              })}
            />
          </div>
        )}
        <Dropzone accept="image/*" multiple={false} preventDropOnDocument onDrop={onDrop} >
          {({getRootProps, getInputProps}) => (
            <div {...getRootProps()} className="dropzone">
              <input {...getInputProps()} />
              Drop image here, or click to choose a file
            </div>
          )}
        </Dropzone>
        <div className="mt-2 custom-control custom-checkbox">
          <input
            type="checkbox"
            className="custom-control-input"
            id="knn-use-spatial-checkbox"
            checked={useSpatial}
            onChange={(e) => setUseSpatial(e.target.checked)}
            disabled
          />
          <label className="custom-control-label" htmlFor="knn-use-spatial-checkbox">
            Use spatial embeddings (slower but more accurate)
          </label>
        </div>
        {size(images) > 0 && <div className="mt-1">
          {isLoading ?
            <Spinner size="sm" color="secondary" /> :
            <Emoji text=":white_check_mark:"/>}&nbsp;&nbsp;&nbsp;
          <span className="text-secondary">Load{isLoading ? "ing" : "ed"} embedding{size(images) > 1 && "s"}</span>
        </div>}
      </PopoverBody>
    </Popover>
  );
};

export default KnnPopover;
