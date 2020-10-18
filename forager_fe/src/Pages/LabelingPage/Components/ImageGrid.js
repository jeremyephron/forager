import React, { useState, useEffect } from "react";
import styled from "styled-components";

const Image = styled.img`
  margin-top: 10px;
  margin-right: 10px;
  object-fit: contain;
  box-shadow: 0 2px 3px -1px rgba(0,0,0,0.5);
  cursor: pointer;
  transition: background 0.2s ease, opacity 0.2s ease;
  &:hover {
    background: white;
    opacity: 0.7;
    mix-blend-mode: multiply;
  }
`;

const Grid = styled.div`
  width: 100%;
  display: flex;
  flex-wrap: wrap;
  height: 100%;
  overflow-y: scroll;
  justify-content: space-evenly;
`;

const ImageGrid = ({
  onImageClick,
  imagePaths,
  imageHeight,
  visibility,
  currentIndex,
  selectedIndices
}) => {
  
  return (
    <Grid>
      {imagePaths.map((path, idx) => <Image key={idx} src={path} onClick={(e) => onImageClick(e, idx)} style={{ height: imageHeight + 'px' , display: (!visibility || visibility[idx] ? "flex" : "none"), "borderStyle": (currentIndex && currentIndex === idx ? "solid" : "none"), "filter": (selectedIndices && selectedIndices.includes(idx) ? "opacity(40%)" : "none")}}/>)}
    </Grid>
  );
}

export default ImageGrid;
