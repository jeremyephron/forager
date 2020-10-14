import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore } from 'redux';

import reducer from './Utils/store.js';
import App from './App';

import "./Constants/styles/global.scss";
import "./Constants/styles/reset.css";

const store = createStore(reducer);

const rootElement = document.getElementById('root');
ReactDOM.render(
  <Provider store={store}>
    <App/>
  </Provider>,
  rootElement
);
