import React, { useState, useEffect } from "react";
import { useHistory } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import styled from "styled-components";

import { colors, baseUrl } from "../../../Constants";
import { Spinner, Button } from "../../../Components";

const Container = styled.div`
  margin: 8vh auto 2vh auto;
`;

const LabelButton = styled(Button)`
    height: 24px;
    font-size: 14px;
    box-shadow: 0 1px 2px 0 rgba(30,54,77,0.50);
`;

const TextHeader = styled.div`
    font-size: 24px;
    font-family: "AirBnbCereal-Book";
`;

const Table = styled.table`
  border-radius: 10px;
  border-spacing: 0;
  border-collapse:separate;
  border: 1px solid ${colors.primary};
  box-shadow: 0 2px 3px 0 rgba(0,0,0,0.20);
  table-layout:fixed;
  overflow: hidden;
  margin-bottom: 25px;
  font-size: 18px;

  tr {
    width: 100%;
  }

  td, th {
    border-bottom: 1px solid lightgray;
    padding: 10px;
    text-align: left;
  }

  tr:last-child > td {
    border-bottom: none
  }

  thead {
    font-family: "AirBnbCereal-Medium";
    color: ${colors.lightText};
    background-color: ${colors.primary};
  }

  tbody {
    font-family: "AirBnbCereal-Book";

      tr:nth-child(odd) {
    background: white;
  }

  tr:nth-child(even) {
    background: rgba(243,246,251,0.8);
  }
`;

const DatasetsTable = ({ datasets, loadingLabels, setLoadingLabels }) => {
  const history = useHistory();
  const dispatch = useDispatch();
  const datasetInfoUrlBase = baseUrl + "/dataset/";
  const downloadLabelsUrlBase = baseUrl + "/dump_annotations/";

  const fetchDatasetInfo = async ( datasetName ) => {
    const datasetInfo = await fetch(datasetInfoUrlBase + datasetName, {
      credentials: 'include',
    }).then(results => results.json());
    dispatch({
      type: 'SET_PREBUILT_INDEX_ID',
      index_id: datasetInfo.indexId,
      dataset: datasetInfo.datasetName,
    });
    history.push('/label', {
      datasetName: datasetInfo.datasetName,
      paths: datasetInfo.paths,
      identifiers: datasetInfo.identifiers
    });
  }
  const downloadLabels = async ( datasetName ) => {
    function forceDownload(blob, filename) {
      var a = document.createElement('a');
      a.download = filename;
      a.href = blob;
      // For Firefox https://stackoverflow.com/a/32226068
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
    setLoadingLabels(true);
    const datasetInfo = await fetch(downloadLabelsUrlBase + datasetName, {
      credentials: 'include',
    }).then(results => results.blob())
    .then(blob => {
      let blobUrl = window.URL.createObjectURL(blob);
      forceDownload(blobUrl, datasetName + '.json');
      setLoadingLabels(false);
    });
  }

  if (loadingLabels) {
    return (
      <div>
      <TextHeader> Downloading labels... </TextHeader>
      <Spinner className='loader' />
      </div>
    )
  } else if (datasets.length > 0) {
    return (
      <Table>
        <thead>
        <tr>
          <th>Name</th>
          <th>No. Images</th>
          <th>No. Labels</th>
          <th>Last Labeled</th>
          <th></th>
          <th></th>
        </tr>
        </thead>
        <tbody>
          {datasets.map((dataset, index) => (
            <tr key={index}>
              <td>{dataset.name}</td>
              <td>{dataset.size}</td>
              <td>{dataset.n_labels}</td>
              <td>{dataset.last_labeled}</td>
              <td>
                <LabelButton alt="true" onClick={() => fetchDatasetInfo(dataset.name)}>
                  Label
                </LabelButton>
              </td>
              <td>
                <LabelButton alt="true" onClick={() => downloadLabels(dataset.name)}>
                  Dump Labels
                </LabelButton>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    );
  } else {
    return (
      <Table>
        <thead>
        <tr>
          <th>Name</th>
          <th>No. Images</th>
          <th>No. Labels</th>
          <th>Last Labeled</th>
          <th></th>
        </tr>
        </thead>
        <tbody>
          <tr>
            <td></td>
          </tr>
        </tbody>
      </Table>
    );
  }
}

const DatasetsTableView = () => {
  const [loadingLabels, setLoadingLabels] = useState(false);

  const datasetsUrl = baseUrl + "/get_datasets";
  const history = useHistory();
  // const [loading, setLoading] = useState(false); // for future use
  const [datasets, setDatasets] = useState([]);

  useEffect(() => {
    fetchDatasets()
  }, []);

  const fetchDatasets = async () => {
    // setLoading(true);
    const newDatasets = await fetch(datasetsUrl, {
      credentials: 'include'
    }).then(results => results.json());
    setDatasets(newDatasets);
    // setLoading(false);
  }

  // if (loading) {
  //   // return spinner
  // }

  return (
    <Container>
      <DatasetsTable datasets={datasets} loadingLabels={loadingLabels} setLoadingLabels={setLoadingLabels}/>
      {!loadingLabels &&
       <Button onClick={() => history.push("/create")}>Create a new dataset</Button>}
    </Container>
  );
}

export default DatasetsTableView;
