import React, { forwardRef, useState } from "react";
import {
  Popover,
  PopoverBody,
} from "reactstrap";

import { Typeahead, useToken, ClearButton } from "react-bootstrap-typeahead";
import cx from "classnames";

import isEqual from "lodash/isEqual";
import uniqWith from "lodash/uniqWith";
import union from "lodash/union";

const FeatureInput = ({ id, features, className, selected, setFeatures, innerRef, ...props }) => {
  return (
    <Typeahead
      id={id}
      className={`typeahead-bar ${className || ""}`}
      options={features}
      selected={selected}
      labelKey="name"
      ref={innerRef}
      {...props}
    />
  );
}

export default FeatureInput;
