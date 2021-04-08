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
  const { disabled, readOnly, active, innerId, onRemove, onValueClick, index } = allProps;

  return (
    <>
      {!disabled && !readOnly ?
        <InteractiveToken {...allProps} ref={ref} id={innerId} /> :
        <StaticToken {...allProps} id={innerId} />}
      <Popover
          placement="top"
          isOpen={isOpen}
          target={innerId}
          trigger="hover"
          toggle={() => setIsOpen(!isOpen)}
          fade={false}
        >
        <PopoverBody>
          {LABEL_VALUES.map(([value, name]) => (
            <div><a href="#" onClick={(e) => onValueClick(value, index, e)} className={`rbt-token ${value}`}>{name.toLowerCase()}</a></div>
          ))}
        </PopoverBody>
      </Popover>
    </>
  );
});

const CategoryInput = ({ id, categories, className, selected, setSelected, setCategories, innerRef, deduplicateByCategory, ...props }) => {
  let options;
  if (deduplicateByCategory) {
    options = categories.flatMap(category =>
      selected.some(s => s.category === category) ?
      [] : [{category, value: LABEL_VALUES[0][0]}]);
  } else {
    options = categories.flatMap(category => {
      for (const [value] of LABEL_VALUES) {
        const proposal = {category, value: value};
        if (!selected.some(s => isEqual(s, proposal))) return [proposal];
      }
      return [];
    });
  }

  const onChange = (selected) => {
    let newSelected = selected.map(
      s => s.customOption ? {category: s.category, value: LABEL_VALUES[0][0]} : s);
    newSelected = uniqWith(newSelected, deduplicateByCategory ?
                           ((a, b) => a.category === b.category) : isEqual)
    setCategories(union(categories, newSelected.map(s => s.category)).sort());
    setSelected(newSelected);
  };

  const onValueClick = (value, index, e) => {
    let newSelected = [...selected];
    newSelected[index] = {...newSelected[index], value};
    onChange(newSelected);
    e.preventDefault();
  };

  const renderToken = (option, { onRemove }, index) => (
    <MyToken
      key={index}
      innerId={`${id}-rbt-token-${index}`}
      className={option.value}
      onRemove={onRemove}
      option={option}
      index={index}
      onValueClick={onValueClick}>
      {option.category}
    </MyToken>
  );

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
