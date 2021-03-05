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
  height: 80vh;
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

  const [confirmIsOpen, setConfirmIsOpen] = useState(false);
  const [confirmCategory, setConfirmCategory] = useState(null);
  const [confirmCategoryIdx, setConfirmCategoryIdx] = useState(null);
  const toggleConfirmIsOpen = (category) => setConfirmIsOpen(!confirmIsOpen);

  let preventCountReload = false;
  useEffect(async () => {
    if (preventCountReload) return;

    setCategories(datasetInfo.categories.slice(0)); // .slice(0) is fastest clone

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

  const setCategoryByIndex = (tag, idx) => {
    if (isReadOnly) return;

    const oldTag = categories[idx];

    categories[idx] = tag;
    setCategories(categories.slice(0));

    delete Object.assign(categoryCounts, {[tag]: categoryCounts[oldTag] })[oldTag];
    setCategoryCounts(categoryCounts);
  };

  const updateCategoryByIndex = async (idx) => {
    const url = new URL(endpoints.updateCategory);
    const body = {
      user: username,
      newCategory: categories[idx],
      oldCategory: datasetInfo.categories[idx],
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    categories.sort()
    setCategories(categories.slice(0));

    preventCountReload = true;
    setDatasetInfo({...datasetInfo, categories: categories});
    preventCountReload = false;
  };

  const deleteCategoryByIndex = async (idx) => {
    if (isReadOnly) return;

    const tag = categories[idx];
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

    categories.splice(idx, 1);
    setCategories(categories);

    delete categoryCounts[tag];
    setCategoryCounts(categoryCounts);

    preventCountReload = true;
    setDatasetInfo({...datasetInfo, categories: categories});
    preventCountReload = false;
  };

  const tableBodyFromTags = () => {
    return (
      <tbody>
        {categories.map((tag, i) => {
          return (
            <tr key={i}>
              <td>
                <input
                  type="text"
                  value={tag}
                  disabled={isReadOnly}
                  onChange={(e) => setCategoryByIndex(e.target.value, i)}
                  onKeyDown={(e) => { if (e.keyCode === 13) e.target.blur(); }}
                  onBlur={(e) => updateCategoryByIndex(i)}
                />
              </td>
              <td>{categoryCounts[tag]}</td>
              <td>
                <Button close disabled={isReadOnly} onClick={(e) => {
                    setConfirmCategory(categories[i]);
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
                <th>Tag</th>
                <th># Images</th>
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