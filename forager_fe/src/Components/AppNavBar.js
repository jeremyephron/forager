import React from "react";
import { NavLink } from "react-router-dom";
import styled from "styled-components";

import ClusterStatus from "./ClusterStatus";

import { colors } from "../Constants";

const Container = styled.div`
  background-color: ${colors.primary};
  align-items: center;
  justify-content: space-between;
  padding: 0px 20px;
  height: 50px;
  position: sticky;
  top: 0;
  border-bottom: 1px solid #22282E;
  box-shadow: 0 2px 5px 0 ${colors.shadow};
  z-index: 1000;
`;

const HomeLink = styled(NavLink)`
  && {
    color: ${colors.lightText};
    font-size: 26px;
    font-family: "Courier";
    text-shadow: 0 2px 3px rgba(0,0,0,0.25);
  }
`;

function AppNavBar() {
  return (
    <Container className="flex-row">
      <HomeLink to="/" activeClassName="home">forager</HomeLink>
      <ClusterStatus />
    </Container>
  );
}

export default AppNavBar;
