import React, {useEffect} from "react";
import useResizeObserver from "use-resize-observer/polyfilled";
import LazyLoad, {forceCheck} from "react-lazyload";

const MARGIN = 3;
const THUMBNAIL_HEIGHT = 200;

const ImageGrid = ({ images, onClick, minRowHeight, imageAspectRatio, selectedPred }) => {
  const { width, height, ref } = useResizeObserver();
  const handleClick = (e, i) => {
    onClick(e, i);
    e.preventDefault();
  }

  const imagesPerRowFloat = width / (minRowHeight * imageAspectRatio + MARGIN);
  const imagesPerRow = Math.floor(imagesPerRowFloat);
  const imageHeight = Math.floor(minRowHeight * imagesPerRowFloat / imagesPerRow);
  const imageWidth = Math.floor(imageAspectRatio * imageHeight);

  useEffect(forceCheck, [images]);

  return (
    <div className="image-grid" ref={ref}>
      {images.map((im, i) =>
        <a href="#" onClick={(e) => handleClick(e, i)}
          style={{width: imageWidth, marginBottom: MARGIN, marginRight: MARGIN}}
        >
          <LazyLoad scrollContainer=".modal" height={imageHeight}>
            <img src={imageHeight > THUMBNAIL_HEIGHT ? im.src : im.thumb}
              className={selectedPred(i) ? "selected" : ""}
            />
          </LazyLoad>
        </a>
      )}
    </div>
  );
}

export default ImageGrid;
