import React, { useEffect } from "react";
import { useSelector, useDispatch } from 'react-redux';
import styled from "styled-components";

import { Button } from "../../../Components";
import { baseUrl } from "../../../Constants";

const BuildButton = styled(Button)`
  font-size: 13px;
  height: 28px;
  width: 200px;
  padding: 0 5px;
  margin-left: 5px;
`;

const API_BASE = baseUrl;
const CREATE_INDEX_ENDPOINT = "/create_index";
const DOWNLOAD_INDEX_ENDPOINT = "/download_index";
const DELETE_INDEX_ENDPOINT = "/delete_index";
const STATUS_POLL_INTERVAL = 3000;  // ms

function BuildIndex({ dataset }) {
  const cluster = useSelector(state => state.cluster);
  const index = useSelector(state => state.indexes[dataset] || {
    id: undefined,
    status: 'INDEX_NOT_BUILT',
  });
  const dispatch = useDispatch();

  let text;
  let enabled = true;

  if (index.id) {
    if (index.status === 'INDEX_READY') {
      text = 'Delete index from disk';
      enabled = false;  // TODO(mihirg): Consider removing this button entirely
    } else if (index.status === 'INDEX_NOT_DOWNLOADED') {
      text = 'Download index to disk';
    } else {
      text = 'Loading index...';
      enabled = false;
    }
  } else {
    text = 'Build index';
    enabled = cluster.status === 'CLUSTER_STARTED';
  }

  const handleClick = async () => {
    if (!enabled) return;

    let endpoint;
    let callback;

    if (index.id) {
      if (index.status === 'INDEX_READY') {
        endpoint = DELETE_INDEX_ENDPOINT;
        callback = 'DELETE_INDEX';
      } else {
        endpoint = DOWNLOAD_INDEX_ENDPOINT;
        callback = 'DOWNLOAD_INDEX';
      }
      endpoint += '/' + index.id;
    } else {
      endpoint = CREATE_INDEX_ENDPOINT + '/' + dataset;
      callback = 'CREATE_INDEX';
    }

    const response = await fetch(API_BASE + endpoint, {
      method: "POST",
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({cluster_id: cluster.id}),
    }).then(response => response.json());
    dispatch({
      type: callback,
      payload: response,
      dataset,
    });
  };

  const updateStatus = async () => {
    if (!index.id) return;

    var url = new URL(API_BASE + '/index/' + index.id);
    url.search = new URLSearchParams({ dataset }).toString();
    const response = await fetch(url, {
      credentials: 'include',
    }).then(response => response.json());
    console.log(response);
    dispatch({
      type: 'SET_INDEX_STATUS',
      payload: response,
      dataset,
    });
  }

  useEffect(() => {
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
