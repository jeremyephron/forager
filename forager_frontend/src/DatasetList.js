import React, { useState, useEffect } from "react";
import { Container } from "reactstrap";
import { Link } from "react-router-dom";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

const endpoints = fromPairs(toPairs({
  getDatasets: "get_datasets",
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const DatasetList = () => {
  const [datasets, setDatasets] = useState([]);
  useEffect(() => {
    async function getDatasetList() {
      const url = new URL(endpoints.getDatasets);
      let _datasets = await fetch(url, {
        method: "GET",
      }).then(r => r.json());
      setDatasets(_datasets.dataset_names);
    }

    getDatasetList();
  }, []);

  return (
    <Container>
      <h2>Forager</h2>
      {datasets.map(d => <div>
        <Link to={`/${d}`} activeClassName="active">{d}</Link>
      </div>)}
    </Container>
  );
};

export default DatasetList;
