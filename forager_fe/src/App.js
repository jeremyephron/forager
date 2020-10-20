import React, { useState, useEffect } from 'react';
import { BrowserRouter, Switch, Route } from "react-router-dom";
import styled from "styled-components";
import { baseUrl } from "./Constants";

import {
  HomePage,
  CreateDatasetPage,
  LabelingPage,
} from "./Pages";

import { AppNavBar, Footer, Button } from "./Components";

async function invalid_ssl_request() {
  var err = false;
  const results = await fetch(baseUrl + "/get_datasets", {
    credentials: 'include'
  }).catch(e => err = true);
  return err;
}

const SSLButtonContainer = styled(Button)`
    display: block;
    width: auto;
    height: auto;
    margin-top: 100px;
    margin-left: auto;
    margin-right: auto;
    font-size: 14px;
    box-shadow: 0 1px 2px 0 rgba(30,54,77,0.50);
`;

function SSLFixButton() {
  return (
    <a href={baseUrl + "/get_datasets"}>
    <SSLButtonContainer>
    SSL certificate issue. Click to fix.
    </SSLButtonContainer>
    </a>
  )
}

const AppContainer = styled.div`
  min-height: 100vh;
  position: relative;
`;

const PageContainer = styled.div`
  padding-bottom: 72px;  /* footer height */
`;

function App() {
  const [hasSSLIssue, updateHasSSLIssue] = useState(true);
  useEffect(() => {
    const getData = async () => {
      updateHasSSLIssue(await invalid_ssl_request());
    };
    getData();
  }, []);
  return (
    <AppContainer>
    <BrowserRouter>
      <AppNavBar />
      <PageContainer>
        {hasSSLIssue
         ? SSLFixButton()
         :
         <Switch>
          <Route exact path="/create" component={CreateDatasetPage} />
          <Route path="/label" render={(props) => (<LabelingPage {...props} />)} />
          <Route exact path="/" component={HomePage} />
         </Switch>}
      </PageContainer>

      <Footer />
    </BrowserRouter>
    </AppContainer>
  );
}

export default App;
