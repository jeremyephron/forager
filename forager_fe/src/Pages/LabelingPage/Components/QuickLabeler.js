import React, { useState, useEffect } from "react";
import styled from "styled-components";

const ImageWindow = styled.img`
  margin-top: 10px;
  margin-right: 10px;
  object-fit: contain;
  box-shadow: 0 2px 3px -1px rgba(0,0,0,0.5);
  transition: background 0.2s ease, opacity 0.2s ease;
`;

const QuickLabeler = ({
  imagePaths,
  currentIndex,
  labels
}) => {
  return (
    <ImageWindow id="quick_labeler" key={currentIndex} src={imagePaths[currentIndex]}>
    </ImageWindow>
  );
}

export default QuickLabeler;
