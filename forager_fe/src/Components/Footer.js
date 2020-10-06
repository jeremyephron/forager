import React from "react";
import styled from "styled-components";

import { colors } from "../Constants";

const Container = styled.div`
  background-color: ${colors.primary};
  align-items: center;
  justify-content: space-between;
  padding: 10px 0 0 0;
  width: 100%;
  height: 36px;
  position: absolute;
  bottom: 0;
  text-align: center;
  box-sizing: border-box;
`;

const CopyrightText = styled.span`
  && {
    color: ${colors.lightText};
    font-size: 11px;
    font-family: "AirBnbCereal-Book";
    vertical-align: middle;
  }
`;

function Footer() {
  return (
    <Container>
      <CopyrightText>
      	Copyright © 2020 Stanford University. All Rights Reserved.
      </CopyrightText>
    </Container>
  );
}

export default Footer;