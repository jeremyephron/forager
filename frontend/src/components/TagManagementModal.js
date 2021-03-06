import React, { useState, useCallback, useEffect, useRef } from "react";
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

const TableContainer = styled.div`
  height: 70vh;
  overflow: auto;  
`;

const TagManagementModal = ({
  isOpen,
  toggle,
  datasetName,
  datasetInfo,
  setDatasetInfo,
  username,
  isReadOnly
}) => {
  // TODO: store with redux
  const [categories, setCategories] = useState([]);
  const [categoryCounts, setCategoryCounts] = useState({});
  const [preventCountReload, setPreventCountReload] = useState(false);

  const kOrderBy = {name: 0, count: 1};
  const kOrderDir = {asc: 0, desc: 1};
  const [orderBy, setOrderBy] = useState(kOrderBy.name);
  const [orderDir, setOrderDir] = useState(kOrderDir.asc);

  const [confirmIsOpen, setConfirmIsOpen] = useState(false);
  const [confirmCategory, setConfirmCategory] = useState(null);
  const [confirmCategoryIdx, setConfirmCategoryIdx] = useState(null);
  const toggleConfirmIsOpen = (category) => setConfirmIsOpen(!confirmIsOpen);

  const sortCategories = (arr) => {
    const copy = arr.slice(0);

    if (orderBy === kOrderBy.name) {
      if (orderDir === kOrderDir.asc) {
        copy.sort((a, b) => a.tag < b.tag ? -1 : 1);
      } else {
        copy.sort((a, b) => b.tag < a.tag ? -1 : 1);
      }
    } else { // orderBy === kOrderBy.count
      if (orderDir === kOrderDir.asc) {
        copy.sort((a, b) => categoryCounts[a.tag] - categoryCounts[b.tag]);
      } else {
        copy.sort((a, b) => categoryCounts[b.tag] - categoryCounts[a.tag]);
      }
    }

    return copy;
  };

  const changeOrdering = (by) => {
    if (by === orderBy) {
      setOrderDir(orderDir === kOrderDir.asc ? kOrderDir.desc : kOrderDir.asc);
    } else {
      setOrderBy(by);
      setOrderDir(kOrderDir.asc)
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
    const oldTag = datasetInfo.categories[srcIdx];
    if (newTag === oldTag) {
      return;
    }

    if (datasetInfo.categories.find((name) => name === newTag) !== undefined) {
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
    const categoriesCopy = categories.map((obj, i) => obj.tag);
    categoriesCopy.sort();
    setDatasetInfo({...datasetInfo, categories: categoriesCopy});

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
    datasetInfo.categories.splice(categories[idx].srcIdx, 1);
    setDatasetInfo({...datasetInfo, categories: datasetInfo.categories.slice(0)});
  };

  useEffect(async () => {
    setCategories(sortCategories(datasetInfo.categories.map(
      (tag, i) => ({tag, srcIdx: i})
    )));

    if (preventCountReload) {
      setPreventCountReload(false);
      return;
    }

    const url = new URL(endpoints.getCategoryCounts + `/${datasetName}`);
    const body = {
      user: username,
      categories: datasetInfo.categories,
    };
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    if (res['numLabeled'] !== undefined) {
      setCategoryCounts(res['numLabeled']);
    }
  }, [datasetInfo.categories]);

  useEffect(() => {
    setCategories(prev => sortCategories(prev))
  }, [categoryCounts, orderBy, orderDir]);

  const tableBodyFromTags = () => {
    return (
      <tbody>
        {categories.map((obj, i) => {
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
              <td>{categoryCounts[datasetInfo.categories[obj.srcIdx]]}</td>
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
    >
      <ModalHeader>Manage Tags</ModalHeader>
      <ModalBody>
        <TableContainer className="pr-2">
          <Table hover borderless size="sm">
            <thead>
              <tr>
                <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy.name)}>
                  Tag Name <FontAwesomeIcon icon={
                    orderBy !== kOrderBy.name ? faSort : (orderDir === kOrderDir.asc ? faSortUp : faSortDown)
                  } />
                </th>
                <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy.count)}>
                  # Images <FontAwesomeIcon icon={
                    orderBy !== kOrderBy.count ? faSort : (orderDir === kOrderDir.asc ? faSortUp : faSortDown)
                  } />
                </th>
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
          cancelBtn={<Button color="secondary" onClick={(e) => toggleConfirmIsOpen()}>Cancel</Button>}
        />
      </ModalBody>
    </Modal>
  );
};

export default TagManagementModal;