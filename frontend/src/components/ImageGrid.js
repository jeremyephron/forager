import React, {useEffect} from "react";
import useResizeObserver from "use-resize-observer/polyfilled";
import LazyLoad, {forceCheck} from "react-lazyload";

const MARGIN = 3;

const ImageGrid = ({ images, onClick, minRowHeight, imageAspectRatio, selectedPred }) => {
  const { width, height, ref } = useResizeObserver();
  const handleClick = (e, i) => {
    onClick(e, i);
    e.preventDefault();
  }

  const imagesPerRowFloat = width / (minRowHeight * imageAspectRatio + MARGIN);
  const imagesPerRow = Math.floor(imagesPerRowFloat);
  const imageWidth = Math.floor(imageAspectRatio * minRowHeight * imagesPerRowFloat / imagesPerRow);

  useEffect(forceCheck, [images]);

  return (
    <div className="image-grid w-100" ref={ref}>
      {images.map((im, i) =>
        <a href="#" onClick={(e) => handleClick(e, i)} style={{width: imageWidth, marginBottom: MARGIN, marginRight: MARGIN}}>
          <LazyLoad scrollContainer=".modal">
            <img src={im.thumb}
              className={selectedPred(i) ? "selected" : ""}
            />
          </LazyLoad>
        </a>
      )}
    </div>
  );
}

export default ImageGrid;
