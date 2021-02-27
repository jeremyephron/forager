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

const endpoints = fromPairs(toPairs({
  updateCategory: 'update_category_v2',
  deleteCategory: 'delete_category_v2',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const TableContainer = styled.div`
  height: 60vh;
  overflow: auto;  
`;

const TagManagementModal = ({
  datasetInfo,
  setDatasetInfo
}) => {
  const [modal, setModal] = useState(false);
  const toggle = () => setModal(!modal);

  const setCategoryByIndex = (tag, idx) => {
    const updateCategoryUrl = new URL(endpoints.updateCategory);
    const body = {
      user: username,
      newCategory: tag,
      oldCategory: datasetInfo.categories[idx]
    };
    const res = fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });

    datasetInfo.categories[idx] = tag;
    setDatasetInfo({...datasetInfo, categories: datasetInfo.categories});
  };

  const deleteCategoryByIndex = async (idx) => {
    const deleteCategoryUrl = new URL(endpoints.deleteCategory);
    const body = {
      user: username,
      category
    };
    const res = fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });

    datasetInfo.categories.splice(idx, 1);
    setDatasetInfo({...datasetInfo, categories: datasetInfo.categories});
  };

  const tableBodyFromTags = () => {
    return (
      <tbody>
        {datasetInfo.categories.map((tag, i) => {
          return (
            <tr key={i}>
              <td>
                <input
                  type="text"
                  value={tag}
                  onChange={(e) => setCategoryByIndex(e.target.value, i)}
                  onKeyDown={(e) => {
                    if (e.keyCode === 13) e.target.blur();
                  }}
                />
              </td>
              <td>0</td>
              <td>
                <Button close onClick={(e) => {
                    deleteCategoryByIndex(i);
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
      isOpen={modal}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
    >
      <ModalHeader toggle={toggle}>Manage Tags</ModalHeader>
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
      </ModalBody>
    </Modal>
  );
};

export default TagManagementModal;