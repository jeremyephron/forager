import React from "react";
import styled from "styled-components";

import { DatasetsTableView } from "./Components";
import { AppNavBar, Footer } from "../../Components";

const Container = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: white;
`;

function HomePage({}) {
  return (
    <Container>
      <DatasetsTableView />
    </Container>
  );
};

export default HomePage;