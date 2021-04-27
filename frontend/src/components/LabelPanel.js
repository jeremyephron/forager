import React, { useState, useEffect } from "react";
import { Typeahead } from "react-bootstrap-typeahead";

import sortBy from "lodash/sortBy";

import NewModeInput from "./NewModeInput";

// TODO(mihirg): Combine with this same constant in other places
const LABEL_VALUES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const LabelPanel = ({
  categories,
  setCategories,
  category,
  setCategory,
}) => {
  const setSelected = (selection) => {
    if (selection.length === 0) {
      setCategory(null);
    } else {
      let c = selection[selection.length - 1];
      if (c.customOption) {  // new
        c = c.label;

        let newCategories = {...categories};
        newCategories[c] = [];  // no custom values to start
        setCategories(newCategories);
      }
      setCategory(c);
    }
  };

  return (
    <div className="d-flex flex-row align-items-center justify-content-between">
      <Typeahead
        multiple
        id="label-mode-bar"
        className="typeahead-bar mr-2"
        options={sortBy(Object.keys(categories), c => c.toLowerCase())}
        placeholder="Category to label"
        selected={category ? [category] : []}
        onChange={setSelected}
        newSelectionPrefix="New category: "
        allowNew={true}
      />
      <div className="text-nowrap">
        {LABEL_VALUES.map(([value, name], i) =>
          <>
            <kbd>{(i + 1) % 10}</kbd> <span className={`rbt-token ${value}`}>{name.toLowerCase()}</span>&nbsp;
          </>)}
        {!!(category) &&
          (categories[category] || []).map((name, i) =>
          <>
            <kbd>{(LABEL_VALUES.length + i + 1) % 10}</kbd> <span className="rbt-token CUSTOM">{name}</span>&nbsp;
          </>)}
        {!!(category) &&
          <NewModeInput
            category={category}
            categories={categories}
            setCategories={setCategories}
          />
        }
      </div>
    </div>
  );
};

export default LabelPanel;
