import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";
import {
  Button,
  Form,
  FormGroup,
  Input,
  Table,
  Modal,
  ModalHeader,
  ModalFooter,
  ModalBody,
} from "reactstrap";

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faSort,
  faSortDown,
  faSortUp
} from "@fortawesome/free-solid-svg-icons";

import styled from "styled-components";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import { ConfirmModal } from "../components";

const endpoints = fromPairs(toPairs({
  updateCategory: 'update_category_v2',
  deleteCategory: 'delete_category_v2',
  getCategoryCounts: 'get_category_counts_v2',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

// TODO(mihirg): Combine with this same constant in other places
const LABEL_VALUES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
  ["CUSTOM", "Other"],
];

const TableContainer = styled.div`
  height: 70vh;
  overflow: auto;
`;

const TagManagementModal = ({
  isOpen,
  toggle,
  datasetName,
  datasetCategories,
  setDatasetCategories,
  username,
  isReadOnly
}) => {
  const categoryList = useMemo(() => Object.keys(datasetCategories), [datasetCategories]);

  // TODO: store with redux
  const [categories, setCategories] = useState([]);
  const [categoryCounts, setCategoryCounts] = useState({});
  const [preventCountReload, setPreventCountReload] = useState(false);

  const kOrderBy = fromPairs(["name", 0].concat(LABEL_VALUES.map(([value], i) => [value, i + 1])));
  const [orderBy, setOrderBy] = useState(kOrderBy.name);
  const [orderAscending, setOrderAscending] = useState(true);

  const [confirmIsOpen, setConfirmIsOpen] = useState(false);
  const [confirmCategory, setConfirmCategory] = useState(null);
  const [confirmCategoryIdx, setConfirmCategoryIdx] = useState(null);
  const toggleConfirmIsOpen = (category) => setConfirmIsOpen(!confirmIsOpen);

  const sortCategories = (arr) => {
    const copy = arr.slice(0);
    const m = orderAscending ? 1 : -1;

    if (orderBy === kOrderBy.name) {
      copy.sort((a, b) => m * (a.tag.toLowerCase() < b.tag.toLowerCase() ? -1 : 1));
    } else {
      for (const [value] of LABEL_VALUES) {
        if (orderBy === kOrderBy[value]) {
          copy.sort((a, b) => m * (categoryCounts[a.tag][value] - categoryCounts[b.tag][value]));
          break;
        }
      }
    }

    return copy;
  };

  const changeOrdering = (by) => {
    if (by === orderBy) {
      setOrderAscending(!orderAscending);
    } else {
      setOrderBy(by);
      setOrderAscending(true);
    }
  };

  const setCategoryByIndex = (tag, idx) => {
    if (isReadOnly) return;

    const oldTag = categories[idx].tag;

    categories[idx].tag = tag;
    setCategories(categories.slice(0));
  };

  const updateCategoryByIndex = async (e, idx, srcIdx) => {
    if (isReadOnly) return;

    const newTag = categories[idx].tag;
    const oldTag = categoryList[srcIdx];
    if (newTag === oldTag) return;

    if (categoryList.find((name) => name === newTag) !== undefined) {
      e.target.setCustomValidity(`"${newTag}" already exists`);
      e.target.reportValidity();
      categories[idx].tag = oldTag;
      setCategories(categories.slice(0));
      return
    }

    const url = new URL(endpoints.updateCategory);
    const body = {
      user: username,
      newCategory: newTag,
      oldCategory: oldTag,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    setPreventCountReload(true);
    const newDatasetCategories = {...datasetCategories};
    newDatasetCategories[newTag] = newDatasetCategories[oldTag];
    delete newDatasetCategories[oldTag];
    setDatasetCategories(newDatasetCategories);

    delete Object.assign(categoryCounts, {[newTag]: categoryCounts[oldTag]})[oldTag];
    setCategoryCounts(categoryCounts);
  };

  const deleteCategoryByIndex = async (idx) => {
    if (isReadOnly) return;

    const tag = categories[idx].tag;
    const url = new URL(endpoints.deleteCategory);
    const body = {
      user: username,
      category: tag,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    delete categoryCounts[tag];
    setCategoryCounts(categoryCounts);

    setPreventCountReload(true);
    let newDatasetCategories = {...datasetCategories};
    delete newDatasetCategories[tag];
    setDatasetCategories(newDatasetCategories);
  };

  useEffect(async () => {
    setCategories(sortCategories(categoryList.map(
      (tag, i) => ({tag, srcIdx: i})
    )));

    if (preventCountReload) {
      setPreventCountReload(false);
      return;
    }

    const url = new URL(endpoints.getCategoryCounts + `/${datasetName}`);
    const body = {
      user: username,
    };
    url.search = new URLSearchParams(body).toString();
    const res = await fetch(url, {
      method: "GET",
    }).then(res => res.json());

    let cats = Object.fromEntries(Object.entries(res).map(([key, val]) => [key, Object.keys(val)]))
    setCategoryCounts(res);
  }, [categoryList]);

  useEffect(() => {
    setCategories(prev => sortCategories(prev))
  }, [categoryCounts, orderBy, orderAscending]);

  const tableBodyFromTags = () => {
    return (
      <tbody>
        {categories.map((obj, i) => {
          const counts = categoryCounts[categoryList[obj.srcIdx]] || {};
          return (
            <tr key={obj.srcIdx}>
              <td>
                <input
                  type="text"
                  value={obj.tag}
                  disabled={isReadOnly}
                  onChange={(e) => setCategoryByIndex(e.target.value, i)}
                  onKeyDown={(e) => { if (e.keyCode === 13) e.target.blur(); }}
                  onBlur={(e) => updateCategoryByIndex(e, i, obj.srcIdx)}
                />
              </td>
              {LABEL_VALUES.map(([value]) => <td>{counts[value] || 0}</td>)}
              <td>
                <Button close disabled={isReadOnly} onClick={(e) => {
                    setConfirmCategory(obj.tag);
                    setConfirmCategoryIdx(i);
                    toggleConfirmIsOpen();
                    document.activeElement.blur();
                  }}
                />
              </td>
            </tr>
          );
        })}
      </tbody>
    );
  }

  return (
    <Modal
      isOpen={isOpen}
      toggle={toggle}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="lg"
    >
      <ModalHeader toggle={toggle}>
        Manage Tags
      </ModalHeader>
      <ModalBody>
        <TableContainer className="pr-2">
          <Table hover borderless size="sm">
            <thead>
              <tr>
                <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy.name)}>
                  Tag Name <FontAwesomeIcon icon={
                    orderBy !== kOrderBy.name ? faSort : (orderAscending ? faSortUp : faSortDown)
                  } />
                </th>
                {LABEL_VALUES.map(([value, name]) => (
                  <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy[value])}>
                    # {name} <FontAwesomeIcon icon={
                      orderBy !== kOrderBy[value] ? faSort : (orderAscending ? faSortUp : faSortDown)
                    } />
                  </th>
                ))}
              </tr>
            </thead>
            {tableBodyFromTags()}
          </Table>
        </TableContainer>

        <ConfirmModal
          isOpen={confirmIsOpen}
          toggle={toggleConfirmIsOpen}
          message={<span>Are you sure you want to delete the tag <strong>{confirmCategory}</strong>? This action cannot be undone.</span>}
          confirmBtn={<Button color="danger" onClick={(e) => {
            deleteCategoryByIndex(confirmCategoryIdx);
            toggleConfirmIsOpen();
          }}>Delete</Button>}
          cancelBtn={<Button color="light" onClick={(e) => toggleConfirmIsOpen()}>Cancel</Button>}
        />
      </ModalBody>
    </Modal>
  );
};

export default TagManagementModal;
