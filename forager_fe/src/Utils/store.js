const initialState = {
  cluster: {
    id: undefined,
    status: 'CLUSTER_NOT_STARTED',
  },
  indexes: {}
}

function reducer(state = initialState, action) {
  switch (action.type) {
    case 'SET_CLUSTER_ID':
      return {...state, cluster: {...state.cluster, id: action.payload.cluster_id}};
    case 'SET_CLUSTER_STATUS': {
      let status;
      if (!action.payload.has_cluster) {
        status = 'CLUSTER_NOT_STARTED';
      } else if (!action.payload.started) {
        status = 'CLUSTER_STARTING';
      } else if (!action.payload.ready) {
        status = 'CLUSTER_PREPARING';
      } else {
        status = 'CLUSTER_STARTED';
      }
      return {...state, cluster: {...state.cluster, status}};
    }
    case 'SET_PREBUILT_INDEX_ID': {
      let indexes = {...state.indexes};
      indexes[action.dataset] = {
        id: action.index_id,
        status: 'INDEX_NOT_DOWNLOADED',
      };
      return {...state, indexes};
    }
    case 'CREATE_INDEX': {
      let indexes = {...state.indexes};
      indexes[action.dataset] = {
        id: action.payload.index_id,
        status: 'INDEX_LOADING',
      };
      return {...state, indexes};
    }
    case 'DOWNLOAD_INDEX': {
      let indexes = {...state.indexes};
      if (action.dataset in indexes) {
        indexes[action.dataset] = {
          id: indexes[action.dataset].id,
          status: 'INDEX_LOADING',
        };
      }
      return {...state, indexes};
    }
    case 'DELETE_INDEX': {
      let indexes = {...state.indexes};
      if (action.dataset in indexes) {
        indexes[action.dataset] = {
          id: indexes[action.dataset].id,
          status: 'INDEX_NOT_DOWNLOADED',
        };
      }
      return {...state, indexes};
    }
    case 'SET_INDEX_STATUS': {
      let indexes = {...state.indexes};
      if (action.dataset in indexes) {
        indexes[action.dataset] = {
          id: indexes[action.dataset].id,
          status: action.payload.has_index ? 'INDEX_READY' : 'INDEX_LOADING',
        };
      }
      return {...state, indexes};
    }
    default:
      return state;
  }
};

export default reducer;
