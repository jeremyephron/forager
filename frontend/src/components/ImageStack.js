import React from "react";
import times from "lodash/times";

const ImageStack = ({ id, onClick, images, showLabel }) => {
  return (
    <a className={`stack ${showLabel ? "" : "nolabel"}`} onClick={onClick}>
      {times(Math.min(4, images.length), (i) =>
        <img key={`stack-${i}`} className="thumb" src={images[i].thumb}></img>
      )}
      {showLabel && <div className="label">
        <b>Cluster {id + 1}</b> ({images.length} image{images.length !== 1 && "s"})
      </div>}
    </a>
  );
}

export default ImageStack;
