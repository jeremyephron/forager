import styled from "styled-components";

import { colors } from "../Constants";

const Button = styled.button`
  width: auto;
  height: 43px;
  border-radius: 8px;
  background: ${props => (props.alt ? colors.alt : colors.primary)};
  font-size: 18px;
  font-family: "AirBnbCereal-Medium";
  color: ${props => (props.alt ? colors.primary : colors.lightText)};
  padding: 0 12px;
  border: 2px solid #13161A;
  box-shadow: 0 2px 3px 0 rgba(30,54,77,0.50);
  transition: background 0.2s ease, box-shadow 0.2s ease;
  cursor: pointer;

  &:hover:enabled {
    background: ${props => (props.alt ? "rgb(245, 250, 254)" : "#454e5a")};
    box-shadow: 0 2px ${props => (props.alt ? "2px" : "6px")} 0 rgba(30,54,77,0.75);
  }

  &:active {
    background: ${props => (props.alt ? "#d8e3ed" : "#323a45")};
    transform: scale(0.98);
  }

  &:focus {
    outline: none;
  }

  &:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }
`;

export default Button;
