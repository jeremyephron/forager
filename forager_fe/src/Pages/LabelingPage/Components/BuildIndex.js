import React, { useEffect, useState } from "react";
import { useSelector, useDispatch } from 'react-redux';
import styled from "styled-components";

import { Button, Select } from "../../../Components";
import { colors, baseUrl } from "../../../Constants";

const BuildButton = styled(Button)`
  font-size: 13px;
  height: 28px;
  width: 150px;
  padding: 0 5px;
  margin-left: 20px;
`;

const API_BASE = baseUrl;
const CREATE_INDEX_ENDPOINT = "/create_index/";
const STATUS_POLL_INTERVAL = 3000;  // ms

function BuildIndex({ dataset }) {
  const cluster = useSelector(state => state.cluster);
  const index = useSelector(state => state.indexes[dataset] || {
    id: undefined,
    status: 'INDEX_NOT_BUILT',
  });
  const dispatch = useDispatch();

  let text = 'Build KNN index';
  let enabled = false;

  if (cluster.status === 'CLUSTER_STARTED') {
    if (index.id) {
      if (index.status === 'INDEX_BUILT') {
        text = 'KNN index built!'
      } else {
        text = 'Building...'
      }
    } else {
      enabled = true;
    }
  }

  const handleClick = async () => {
    if (!enabled) return;

    const response = await fetch(API_BASE + CREATE_INDEX_ENDPOINT + dataset, {
      method: "POST",
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({cluster_id: cluster.id})
    }).then(response => response.json());
    dispatch({
      'type': 'SET_INDEX_ID',
      'dataset': dataset,
      'payload': response,
    });
  };

  const updateStatus = async () => {
    if (!index.id) return;

    const response = await fetch(API_BASE + '/index/' + index.id, {
      credentials: 'include',
    }).then(response => response.json());
    console.log(response);
    dispatch({
      'type': 'SET_INDEX_STATUS',
      'dataset': dataset,
      'payload': response,
    });
  }

  useEffect(() => {
    updateStatus();
    if (index.status.endsWith('ING')) {
      const interval = setInterval(updateStatus, STATUS_POLL_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [index.id, index.status]);

  return (
    <BuildButton disabled={!enabled} onClick={handleClick}>
      {text}
    </BuildButton>
  );
}

export default BuildIndex;
