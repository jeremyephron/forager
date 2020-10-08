import React from "react";
import { withRouter } from 'react-router-dom';
import styled from "styled-components";

import { Button, Spinner } from "../../Components";
import { colors } from "../../Constants";
import { sleep } from "../../Utils";

const Container = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: white;
`;

const SubContainer = styled.div`
  margin: 8vh auto;
`;

const TitleHeader = styled.h1`
  font-family: "AirBnbCereal-Medium";
  font-size: 24px;
  color: ${colors.primary};
  margin-bottom: 1.5em;
  text-align: center;
`;

const Form = styled.form`
  font-family: "AirBnbCereal-Book";
  font-size: 18px;

  width: 80vw;
  max-width: 400px;
  input {
    margin-top: 0.5em;
    margin-bottom: 1em;
    padding-left: 5px;
    width: 100%;
    border-radius: 5px;
    border: 1px solid gray;
    height: 20px;
    font-size: 16px;
  }
`;

const SubmitButton = styled(Button)`
  font-size: 16px;
  height: 34px;
`;

class NewDatasetForm extends React.Component {
  constructor(props) {
    super(props);
    this.createDatasetUrl = "https://127.0.0.1:8000/api/create_dataset";
    this.state = {loading: false};

    this.handleSubmit = this.handleSubmit.bind(this);
  }

  async handleSubmit(event) {
    event.preventDefault();
    this.setState({loading: true});
    const data = {dataset: event.target.dataset.value, dirpath: event.target.dirpath.value};
    const response = await fetch(this.createDatasetUrl, {
      method: "POST",
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    .then(response => response.json());

    if (response.status === 'success') {
      this.setState({loading: false});
      this.props.history.push('/label', {
          datasetName: response.datasetName,
          paths: response.paths,
          identifiers: response.identifiers
      });
    } else {
      await sleep(750);
      this.setState({loading: false});
      alert(response.message);
    }
  }

  render() {
    if (this.state.loading) {
      return (<Spinner className='loader' />);
    } else {
      return (
        <Form onSubmit={this.handleSubmit}>
        <label htmlFor="dataset">Name</label><br/>
        <input onChange={this.handleChange} required type="input" id="dataset" name="dataset" placeholder="name of the dataset (no spaces!)" pattern="^\S+$"/><br/>
        <label htmlFor="dirpath">Path to directory</label><br/>
        <input required type="input" id="dirpath" name="dirpath" placeholder="full path to GCS bucket" /><br/><br/>
        <SubmitButton type="submit">Create dataset</SubmitButton>
        </Form>
      );
    }
  }
}

const NewDatasetFormWithRouter = withRouter(NewDatasetForm);

function CreateDatasetPage() {
  return (
    <Container>
    <SubContainer>
      <TitleHeader>Create a New Dataset</TitleHeader>
      <NewDatasetFormWithRouter />
    </SubContainer>
    </Container>
  );
};

export default CreateDatasetPage;
