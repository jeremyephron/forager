import React from "react";
import { Typeahead } from "react-bootstrap-typeahead";

const FeatureInput = ({ features, className, selected, setSelected, ...props }) => {
  return (
    <Typeahead
      className={`typeahead-bar ${className || ""}`}
      options={features}
      selected={selected}
      onChange={setSelected}
      labelKey="name"
      {...props}
    />
  );
}

export default FeatureInput;
