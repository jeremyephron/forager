import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

import {
  BrowserRouter as Router,
  Switch,
  Route,
  Redirect,
} from "react-router-dom";

import TimeAgo from "javascript-time-ago";
import en from "javascript-time-ago/locale/en";
TimeAgo.addDefaultLocale(en);

const DEFAULT_DATASET_NAME = "waymo";

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <Switch>
        <Redirect exact from="/" to={`/${DEFAULT_DATASET_NAME}`} />
        <Route path="/:datasetName" children={<App />} />
      </Switch>
    </Router>
  </React.StrictMode>,
  document.getElementById('root')
);
