import React from "react";
import { Typeahead } from "react-bootstrap-typeahead";

const FeatureInput = ({ id, features, className, selected, setSelected, placeholder }) => {
  return (
    <Typeahead
      id={id}
      className={`typeahead-bar ${className || ""}`}
      options={features}
      selected={selected}
      onChange={setSelected}
      labelKey="name"
      placeholder={placeholder}
    />
  );
}

export default FeatureInput;
