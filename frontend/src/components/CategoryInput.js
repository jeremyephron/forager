import React, { forwardRef, useState, useMemo } from "react";
import {
  Popover,
  PopoverBody,
} from "reactstrap";

import { Typeahead, useToken, ClearButton } from "react-bootstrap-typeahead";
import cx from "classnames";

import isEqual from "lodash/isEqual";
import sortBy from "lodash/sortBy";
import uniqWith from "lodash/uniqWith";
import union from "lodash/union";

import NewModeInput from "./NewModeInput";

// TODO(mihirg): Combine with this same constant in other places
const LABEL_VALUES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const InteractiveToken = forwardRef((
  { active, children, className, onRemove, tabIndex, ...props },
  ref
) => (
  <div
    {...props}
    className={cx('rbt-token', 'rbt-token-removeable', {
      'rbt-token-active': !!active,
    }, className)}
    ref={ref}
    tabIndex={tabIndex || 0}>
    {children}
    <ClearButton
      className="rbt-token-remove-button"
      label="Remove"
      onClick={onRemove}
      tabIndex={-1}
    />
  </div>
));

const StaticToken = ({ children, className, disabled, href }) => {
  const classnames = cx('rbt-token', {
    'rbt-token-disabled': disabled,
  }, className);

  if (href && !disabled) {
    return (
      <a className={classnames} href={href}>
        {children}
      </a>
    );
  }

  return (
    <div className={classnames}>
      {children}
    </div>
  );
};

const MyToken = forwardRef((props, ref) => {
  const [isOpen, setIsOpen] = useState(false);
  const allProps = useToken(props);
  const {
    disabled,
    active,
    innerId,
    onRemove,
    onValueClick,
    index,
    customValues,
    category,
    categories,
    setCategories,
    allowNewModes,
    deduplicateByCategory,
    populateAll,
  } = allProps;

  const target = document.getElementById(innerId);

  return (
    <>
      {!disabled ?
        <InteractiveToken {...allProps} ref={ref} id={innerId} /> :
        <StaticToken {...allProps} id={innerId} />}
      {target !== null && <Popover
        placement="top"
        isOpen={isOpen && !!(document.getElementById(innerId))}
        target={innerId}
        trigger="hover"
        toggle={() => setIsOpen(!isOpen)}
        fade={false}
      >
        <PopoverBody onClick={e => e.stopPropagation()}>
          {LABEL_VALUES.map(([value, name]) =>
            <div>
              <a
                href="#"
                onClick={(e) => onValueClick(value, index, e)}
                className={`rbt-token ${value}`}
              >
                {name.toLowerCase()}
              </a>
            </div>)}
          {customValues.map(name =>
            <div>
              <a
                href="#"
                onClick={(e) => onValueClick(name, index, e)}
                className="rbt-token CUSTOM"
              >
                {name.toLowerCase()}
              </a>
            </div>)}
          {allowNewModes && <div>
            <NewModeInput
              category={category}
              categories={categories}
              setCategories={setCategories}
            />
          </div>}
          {!deduplicateByCategory && <div>
            <a
              href="#"
              onClick={(e) => populateAll(index, e)}
              className="rbt-token ALL"
            >
              (All of the above)
            </a>
          </div>}
        </PopoverBody>
      </Popover>}
    </>
  );
});

const CategoryInput = ({ id, categories, className, selected, setSelected, setCategories, innerRef, deduplicateByCategory, allowNewModes, ...props }) => {
  let options;

  const sortedCategories = useMemo(() => sortBy(Object.entries(categories), ([c]) => c.toLowerCase()), [categories]);

  if (deduplicateByCategory) {
    options = sortedCategories.flatMap(([category]) =>
      selected.some(s => s.category === category) ?
      [] : [{category, value: LABEL_VALUES[0][0]}]);
  } else {
    options = sortedCategories.flatMap(([category, custom_values]) => {
      for (const value of [...LABEL_VALUES, ...custom_values]) {
        const proposal = {category, value: Array.isArray(value) ? value[0] : value};
        if (!selected.some(s => isEqual(s, proposal))) return [proposal];
      }
      return [];
    });
  }

  const onChange = (selected) => {
    let newSelected = selected.map(s => {
      if (s.customOption) {  // added
        let newCategories = {...categories};
        newCategories[s.category] = [];  // no custom values to start
        setCategories(newCategories);

        return {category: s.category, value: LABEL_VALUES[0][0]};
      }
      return s;
    });
    newSelected = uniqWith(newSelected, deduplicateByCategory ?
                           ((a, b) => a.category === b.category) : isEqual);
    setSelected(newSelected);
  };

  const onValueClick = (value, index, e) => {
    let newSelected = [...selected];
    newSelected[index] = {...newSelected[index], value};
    onChange(newSelected);
    e.preventDefault();
  };

  const populateAll = (index, e) => {
    const category = selected[index].category;
    const before = selected.slice(0, index);
    const standard = LABEL_VALUES.map(([value]) => {return {category, value};});
    const custom = (categories[category]).map(value => {return {category, value};});
    const after = selected.slice(index + 1);
    onChange([...before, ...standard, ...custom, ...after]);
    e.preventDefault();
  };

  const renderToken = (option, { onRemove, disabled, key }, index) => {
    const isCustom = !LABEL_VALUES.some(([value]) => value === option.value);
    return (
      <MyToken
        key={key}
        disabled={disabled}
        innerId={`${id}-rbt-token-${index}`}
        className={isCustom ? "CUSTOM" : option.value}
        onRemove={onRemove}
        option={option}
        index={index}
        onValueClick={onValueClick}
        populateAll={populateAll}
        customValues={categories[option.category] || []}
        category={option.category}
        categories={categories}
        setCategories={setCategories}
        allowNewModes={allowNewModes}
        deduplicateByCategory={deduplicateByCategory}>
        {option.category}{isCustom ? ` (${option.value})` : ""}
      </MyToken>
    );
  };

  return (
    <Typeahead
      id={id}
      multiple
      className={`typeahead-bar ${className || ""}`}
      options={options}
      selected={selected}
      onChange={onChange}
      renderToken={renderToken}
      labelKey="category"
      newSelectionPrefix="New category: "
      ref={innerRef}
      {...props}
    />
  );
}

export default CategoryInput;
