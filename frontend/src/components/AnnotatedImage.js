import React, {useEffect, useState} from "react";
import Dimensions from "react-dimensions";
import {Stage, Layer, Image, Line} from "react-konva";
import useImage from "use-image";



const AnnotatedImage = ({ url, boxes, onClick, containerWidth, containerHeight }) => {
  console.log(url);

  const [image] = useImage(url);

  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [scale, setScale] = useState(1);

  console.log(url, width);

  useEffect(() => {
    if (!image) {
      return;
    }
    console.log(url);
    const aspectRatio = image.width / image.height;
    //const scale = Math.min(width / image.width, height / image.height);
    const width = containerWidth;
    const height = width / aspectRatio;
    const scale = width / image.width;

    setWidth(width);
    setHeight(height);
    setScale(scale);
  }, [image]);


  return (
    <Stage width={width} height={height}>
      <Layer>
        <Image image={image} width={width} height={height} />
      </Layer>
      <Layer>
        {boxes.map(box => {
          const inv_s = 1.0 / scale;
          const x1 = box.x1 * inv_s;
          const y1 = box.y1 * inv_s;
          const x2 = box.x2 * inv_s;
          const y2 = box.y2 * inv_s;
          return (
            <Line
              points={[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}
              stroke='red'
              strokeWidth={1}
              closed
            />
          );
        })}
      </Layer>
    </Stage>
  );
}

export default Dimensions({elementResize: true})(AnnotatedImage);
