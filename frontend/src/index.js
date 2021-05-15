import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

import {
  BrowserRouter as Router,
  Switch,
  Route,
} from "react-router-dom";

import TimeAgo from "javascript-time-ago";
import en from "javascript-time-ago/locale/en";
TimeAgo.addDefaultLocale(en);

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <Switch>
        <Route path="/:datasetName" children={<App />} />
      </Switch>
    </Router>
  </React.StrictMode>,
  document.getElementById('root')
);
