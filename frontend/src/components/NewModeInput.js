import React, { useState } from "react";
import {
  Input,
} from "reactstrap";

import uniq from "lodash/uniq";

const MAX_CUSTOM_CATEGORIES = 6;

const NewModeInput = ({category, categories, setCategories}) => {
  const [value, setValue] = useState("");

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      const newValue = value.trim().toLowerCase();
      if (newValue !== "") {
        let newCategories = {...categories};
        newCategories[category] = uniq([...newCategories[category], newValue]);
        setCategories(newCategories);
      }
      setValue("");
    }
    e.stopPropagation();
  };

  return categories[category].length < MAX_CUSTOM_CATEGORIES ? (
    <Input
      bsSize="sm"
      placeholder="New mode"
      className="new-mode-input"
      value={value}
      onChange={e => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      onFocus={e => e.stopPropagation()}/>
  ) : <></>;
};

export default NewModeInput;
