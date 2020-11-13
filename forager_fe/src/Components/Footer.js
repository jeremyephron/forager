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

function Footer() {
  return (
    <Container>
    </Container>
  );
}

export default Footer;