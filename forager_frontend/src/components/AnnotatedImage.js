import React, {useEffect, useState} from "react";
import Dimensions from "react-dimensions";
import {Stage, Layer, Image, Line, Text} from "react-konva";
import Konva from "konva";
import useImage from "use-image";


const AnnotatedImage = ({ url, boxes, onClick, containerWidth, containerHeight }) => {
  const [image] = useImage(url);

  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    if (!image) {
      return;
    }
    const aspectRatio = image.width / image.height;
    //const scale = Math.min(width / image.width, height / image.height);
    const width = containerWidth;
    const height = width / aspectRatio;
    const newScale = width / image.width;

    setWidth(width);
    setHeight(height);
    setScale(newScale);
  }, [image, containerWidth]);


  return (
    <Stage width={width} height={height}>
      <Layer>
        <Image image={image} width={width} height={height} />
      </Layer>
      <Layer>
        {boxes.map(box => {
          const x1 = Math.round(box.x1 * scale);
          const y1 = Math.round(box.y1 * scale);
          const x2 = Math.round(box.x2 * scale);
          const y2 = Math.round(box.y2 * scale);

          const label = new Konva.Text({
            text: box.category, fontsize: 12
          })
          const {width: textWidth, height: textHeight} = label.measureSize()
          return (<>
            <Line
              points={[x1, y1, x2, y1, x2, y2, x1, y2]}
              stroke='red'
              strokeWidth={1}
              closed
            />
            <Text
              x={x2-textWidth*2}
              y={y2-textHeight}
              text={box.category}
              fontsize={12}
              fill='red'
            />
          </>
          );
        })}
      </Layer>
    </Stage>
  );
}

export default Dimensions({elementResize: true})(AnnotatedImage);
