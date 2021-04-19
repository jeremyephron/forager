import React, { useState } from "react";
import {
  Popover,
  PopoverBody,
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
    let promises = [];
    for (const file of acceptedFiles) {
      const uuid = uuidv4();
      dispatch({
        type: "ADD_IMAGE_FILE",
        file,
        uuid,
      });
      promises.push(generateEmbedding({image_data: await toBase64(file)}, uuid));
    }
    await Promise.all(promises);
  };

  const isLoading = some(Object.values(images).map(i => !(i.embedding)));

  return (
    <Popover
      placement="bottom"
      isOpen={true}
      target="ordering-mode"
      trigger="hover"
      toggle={() => setIsOpen(!isOpen)}
      fade={false}
      popperClassName={`knn-popover ${(isOpen || isLoading || hasDrag) ? "visible" : "invisible"}`}
    >
      <PopoverBody>
        {Object.entries(images).map(([uuid, image]) =>
          <div className="removable-image mb-2">
            <img
              key={uuid}
              src={image.src}
            />
            <FontAwesomeIcon
              icon={faTimesCircle}
              className="remove-icon"
              onClick={() => dispatch({
                type: "DELETE_IMAGE",
                uuid,
              })}
            />
          </div>
        )}
        <Dropzone accept="image/*" multiple preventDropOnDocument onDrop={onDrop} >
          {({getRootProps, getInputProps}) => (
            <div {...getRootProps()} className="dropzone mb-2">
              <input {...getInputProps()} />
              Drop image here, or click to choose a file
            </div>
          )}
        </Dropzone>
        <div className="custom-control custom-checkbox">
          <input
            type="checkbox"
            className="custom-control-input"
            id="knn-use-spatial-checkbox"
            checked={useSpatial}
            onChange={(e) => setUseSpatial(e.target.checked)}
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
