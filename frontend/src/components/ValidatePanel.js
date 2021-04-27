import React, { useState, useEffect } from "react";
import {
  Button,
} from "reactstrap";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import FeatureInput from "./FeatureInput";

const endpoints = fromPairs(toPairs({
  // addAnnotationsToResultSet: 'add_annotations_to_result_set_v2',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const ValidatePanel = ({
  modelInfo,
  isVisible,
}) => {
  const [validateModel, setValidateModel] = useState(null);

  if (!isVisible) return null;
  return (
    <div>
      <div className="d-flex flex-row align-items-center justify-content-between">
        <FeatureInput
          id="validate-model-bar"
          className="mr-2"
          placeholder="Model to validate"
          features={modelInfo.filter(m => m.with_output)}
          selected={validateModel}
          setSelected={setValidateModel}
        />
        <Button
          color="light"
          // onClick={startDnnInference}
          disabled={!!!(validateModel)}
        >Start validation</Button>
      </div>
    </div>
  );
};

export default ValidatePanel;
