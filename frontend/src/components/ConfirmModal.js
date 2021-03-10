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

const ConfirmModal = ({
  isOpen,
  toggle,
  message,
  confirmBtn,
  cancelBtn,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
    >
      <ModalHeader>Confirmation Required</ModalHeader>
      <ModalBody>{message}</ModalBody>
      <ModalFooter>
        {confirmBtn}{" "}
        {cancelBtn}
      </ModalFooter>
    </Modal>
  );
};

export default ConfirmModal;