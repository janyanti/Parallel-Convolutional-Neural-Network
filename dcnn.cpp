class sample_t {
  private:
    float **x;
    int **y;
}

class model {
  private:
    int num_layers;
    int *num_units; 
    // each layer has a 2d set of weights
    float ***weights;
    float learning_rate;
    int num_epochs;

    int num_samples;
    int input_rows; 
    int input_cols; // should be 1 
    int output_rows; 
    int output_cols; // should be 1
    sample_t *samples;
    
  public: 
    void train();
}


void model::train() {
  // iterators 
  int w, e, s, i;
  // temp calculations 
  float **temp[num_layers];
  float **final[num_layers];
  
  // alloc weights 
  int rows, cols;
  for (i = 0; i < num_layers; i++) {
    int rows = num_units[i];
    if (i == num_layers - 1) {
      rows = output_rows;
    } else {
      rows = num_units[i];
    }

    if (i == 0) {
      cols = input_rows;
    } else {
      cols = num_units[i-1] + 1;
    }

    weights[i] = float_alloc_2d(rows, cols);
  }

  for (e = 0; e < num_epochs; e++) {
    for (s = 0; s < num_samples; s++) {
      sample_t s = samples[s];
      // forward computation
      for (i = 0; i < num_layers; i++) {
        temp[i] = weights[i] * sample.x;
        final[i] = append(ones,temp[i]);
      }
      // backward computation
      for (i = num_layers - 1; i > 0; i--) {
        
      }
      // update all the weights
    }
  }
}