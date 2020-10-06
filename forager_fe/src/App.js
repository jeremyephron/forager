import React from 'react';
import { BrowserRouter, Switch, Route } from "react-router-dom";
import styled from "styled-components";

import {
  HomePage,
  CreateDatasetPage,
  LabelingPage,
} from "./Pages";

import { AppNavBar, Footer } from "./Components";

const AppContainer = styled.div`
  min-height: 100vh;
  position: relative;
`;

const PageContainer = styled.div`
  padding-bottom: 72px;  /* footer height */
`;

function App() {
  return (
    <AppContainer>
    <BrowserRouter>
      <AppNavBar />

      <PageContainer>
        <Switch>
          <Route exact path="/create" component={CreateDatasetPage} />
          <Route path="/label" render={(props) => (<LabelingPage {...props} />)} />
          <Route exact path="/" component={HomePage} />
        </Switch>
      </PageContainer>

      <Footer />
    </BrowserRouter>
    </AppContainer>
  );
}

export default App;
