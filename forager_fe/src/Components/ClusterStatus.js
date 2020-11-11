import React, { useEffect, useState } from "react";
import { useSelector, useDispatch } from 'react-redux';
import styled from "styled-components";

import { colors, baseUrl } from "../Constants";

const StatusButton = styled.a`
  float: right;
  font-family: "AirBnbCereal-Book";
  color: ${colors.lightText};
  display: flex;
  height: 100%;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  padding: 0 10px;
  transition: background 0.2s ease;

  &:hover:not(.disabled) {
    cursor: pointer;
    background: #454e5a;
  }
`;

const SubText = styled.div`
  font-size: 14px;
  line-height: 20px;
`;

const API_BASE = baseUrl;
const START_CLUSTER_ENDPOINT = "/start_cluster";
const STOP_CLUSTER_ENDPOINT = "/stop_cluster";
const STATUS_POLL_INTERVAL = 3000;  // ms

function ClusterStatus() {
  const cluster = useSelector(state => state.cluster);
  const dispatch = useDispatch();

  let mainText, subText;
  switch (cluster.status) {
    case 'CLUSTER_STARTING':
      mainText = 'Cluster starting...';
      break;
    case 'CLUSTER_PREPARING':
      mainText = 'Cluster preparing...';
      break;
    case 'CLUSTER_STARTED':
      mainText = 'Cluster ready';
      subText = '(Click to shut down)';
      break;
    default:
      mainText = 'Cluster not started';
      subText = '(Click to start)';
  }

  const handleClick = async () => {
    let endpoint;
    if (cluster.status === 'CLUSTER_NOT_STARTED') {
      endpoint = START_CLUSTER_ENDPOINT;
    } else if (cluster.status === 'CLUSTER_STARTED') {
      endpoint = STOP_CLUSTER_ENDPOINT + '/' + cluster.id;
    } else {
      return;
    }

    const response = await fetch(API_BASE + endpoint, {
      method: "POST",
      credentials: 'include',
    }).then(response => response.json());
    dispatch({
      type: 'SET_CLUSTER_ID',
      payload: response,
    });
  };

  const updateStatus = async () => {
    const response = await fetch(API_BASE + '/cluster/' + cluster.id, {
      credentials: 'include',
    }).then(response => response.json());
    dispatch({
      type: 'SET_CLUSTER_STATUS',
      payload: response,
    });
  }

  useEffect(() => {
    updateStatus();
    if (cluster.status.endsWith('ING') || cluster.status === 'CLUSTER_STARTED') {
      const interval = setInterval(updateStatus, STATUS_POLL_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [cluster.id, cluster.status]);

  return (
    <StatusButton className={subText ? '' : 'disabled'} onClick={handleClick}>
      <div>{mainText}</div>
      <SubText>{subText}</SubText>
    </StatusButton>
  );
}

export default ClusterStatus;
