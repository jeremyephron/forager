import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { Button, Select } from "../../../Components";

const RowContainer = styled.div`
  display: flex;
  flex-direction: row;
  background-color: white;
`;

const ColContainer = styled.div`
  display: flex;
  flex-direction: column;
  background-color: white;
  margin-right: 10px;
`;

const FetchButton = styled(Button)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const OptionsSelect = styled(Select)`
  font-size: 13px;
  height: 28px;
  padding: 0 5px;
`;

const TrainProgress = ({
  onTrain,
  AP,
  F1,
  CurrEpoch,
  TotalEpochs
}) => {
  
  return (
    <ColContainer id="train_progress">
      <RowContainer>
        <input placeholder="Train Category"/>
        <input placeholder="Val Category"/>
        <OptionsSelect alt="true" id="select_model_type">
          <option value="svm">SVM</option>
          <option value="nn">NN</option>
        </OptionsSelect>
        <input placeholder="Output Category"/>
        <FetchButton id="train_button" onClick={onTrain}>Train</FetchButton>
      </RowContainer>
      <RowContainer>
        <ColContainer>
          <div>Validation Performance</div>
          <div>AP: {AP} </div>
          <div>Image F_1: {F1} </div>
        </ColContainer>
        <ColContainer>
          <div>Train Progress</div>
          <div>Epoch: {CurrEpoch}/{TotalEpochs} </div>
        </ColContainer>
      </RowContainer>
    </ColContainer>
  );
}

export default TrainProgress;
