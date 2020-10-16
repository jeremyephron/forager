import React, { useState, useEffect } from "react";
import { useHistory } from 'react-router-dom';
import styled from "styled-components";

import { colors, baseUrl } from "../../../Constants";
import { Button } from "../../../Components";

const Container = styled.div`
  margin: 8vh auto 2vh auto;
`;

const LabelButton = styled(Button)`
    height: 24px;
    font-size: 14px;
    box-shadow: 0 1px 2px 0 rgba(30,54,77,0.50);
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

const DatasetsTable = ({ datasets }) => {
  const history = useHistory();
  const datasetInfoUrlBase = baseUrl + "/dataset/";

  const fetchDatasetInfo = async ( datasetName ) => {
    const datasetInfo = await fetch(datasetInfoUrlBase + datasetName, {
      credentials: 'include',
    }).then(results => results.json());
    history.push('/label', {
      datasetName: datasetInfo.datasetName,
      paths: datasetInfo.paths,
      identifiers: datasetInfo.identifiers
    });
  }

  if (datasets.length > 0) {
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
      <DatasetsTable datasets={datasets} />
      <Button onClick={() => history.push("/create")}>Create a new dataset</Button>
    </Container>
  );
}

export default DatasetsTableView;
